from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from feature_selection import choose_volatility_column, ensure_core_features, select_hmm_feature_matrix
from regime_utils import semantic_volatility_labels, summarize_regimes, transition_table


INSTALL_HINT = "Install hmmlearn with: pip install hmmlearn"


def _hmm_parameter_count(n_states: int, n_features: int, covariance_type: str) -> int:
    transition_params = n_states * (n_states - 1)
    start_params = n_states - 1
    mean_params = n_states * n_features
    if covariance_type == "full":
        covariance_params = n_states * n_features * (n_features + 1) // 2
    else:
        covariance_params = n_states * n_features
    return int(transition_params + start_params + mean_params + covariance_params)


def _fit_candidate_models(
    X: np.ndarray,
    states: list[int],
    config: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:
        raise RuntimeError(f"hmmlearn is not available. {INSTALL_HINT}. Import error: {exc}") from exc

    hmm_cfg = config.get("hmm", {})
    covariance_type = str(hmm_cfg.get("covariance_type", "diag"))
    model_rows = []
    best_model = None
    best_bic = np.inf
    for n_states in states:
        t0 = time.perf_counter()
        row = {
            "n_states": int(n_states),
            "status": "failed",
            "log_likelihood": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "runtime_seconds": np.nan,
            "error": None,
        }
        try:
            model = GaussianHMM(
                n_components=int(n_states),
                covariance_type=covariance_type,
                n_iter=int(hmm_cfg.get("n_iter", 100)),
                tol=float(hmm_cfg.get("tol", 0.001)),
                random_state=int(config.get("seed", 20260609)),
            )
            model.fit(X)
            log_likelihood = float(model.score(X))
            params = _hmm_parameter_count(int(n_states), X.shape[1], covariance_type)
            aic = 2 * params - 2 * log_likelihood
            bic = params * np.log(max(X.shape[0], 1)) - 2 * log_likelihood
            row.update(
                {
                    "status": "success",
                    "log_likelihood": log_likelihood,
                    "aic": float(aic),
                    "bic": float(bic),
                    "runtime_seconds": float(time.perf_counter() - t0),
                }
            )
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except Exception as exc:
            row["error"] = repr(exc)
            row["runtime_seconds"] = float(time.perf_counter() - t0)
        model_rows.append(row)
    model_selection = pd.DataFrame(model_rows)
    if best_model is None:
        raise RuntimeError("All candidate HMM fits failed.")
    return best_model, model_selection


def run_hmm_regime_detection(
    df: pd.DataFrame,
    asset: str,
    frequency: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hmm_cfg = config.get("hmm", {})
    min_rows = int(hmm_cfg.get("min_rows", 250))
    prepared = ensure_core_features(df, rolling_window=int(config.get("quantile", {}).get("default_rolling_window", 60)))
    X, scaled, feature_columns, feature_diagnostics = select_hmm_feature_matrix(prepared, config)
    if len(scaled) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows for HMM; got {len(scaled)} for {asset} {frequency}.")

    states = [int(item) for item in hmm_cfg.get("states", [2, 3, 4])]
    model, model_selection = _fit_candidate_models(X, states, config)
    n_states = int(model.n_components)
    raw_states = model.predict(X)
    posterior = model.predict_proba(X)
    confidence = posterior.max(axis=1)

    vol_col = choose_volatility_column(prepared)
    vol_values = pd.to_numeric(prepared[vol_col], errors="coerce") if vol_col else pd.Series(np.nan, index=prepared.index)
    state_vol = pd.Series(vol_values.to_numpy(), index=pd.Index(raw_states)).groupby(level=0).mean().sort_values()
    labels = semantic_volatility_labels(n_states)
    mapping = {int(raw_state): labels[i] for i, raw_state in enumerate(state_vol.index.tolist())}
    semantic = pd.Series(raw_states).map(mapping).fillna("unknown").astype(str)
    regime_method = f"hmm_{n_states}_states"

    labels_df = pd.DataFrame(
        {
            "timestamp": prepared["timestamp"].reset_index(drop=True),
            "asset": asset,
            "frequency": frequency,
            "regime_method": regime_method,
            "regime_label": semantic,
            "raw_state": raw_states.astype(int),
            "regime_confidence": confidence,
            "selected_n_states": n_states,
            "feature_set": ",".join(feature_columns),
        }
    )
    for state in range(posterior.shape[1]):
        labels_df[f"posterior_state_{state}"] = posterior[:, state]

    working = prepared.reset_index(drop=True).copy()
    working["regime_label"] = semantic.values
    summary = summarize_regimes(working, "regime_label", asset, frequency, regime_method, vol_col=vol_col)
    transitions = transition_table(working, "regime_label", asset, frequency, regime_method)

    persistence_rows = []
    if hasattr(model, "transmat_"):
        semantic_state_for_raw = pd.Series(mapping, name="regime_label")
        for raw_state in range(model.transmat_.shape[0]):
            self_prob = float(model.transmat_[raw_state, raw_state])
            persistence_rows.append(
                {
                    "asset": asset,
                    "frequency": frequency,
                    "regime_method": regime_method,
                    "raw_state": int(raw_state),
                    "regime_label": semantic_state_for_raw.get(raw_state, "unknown"),
                    "self_transition_probability": self_prob,
                    "expected_duration_observations": float(1.0 / max(1.0 - self_prob, 1e-12)),
                }
            )
    persistence = pd.DataFrame(persistence_rows)

    model_selection.insert(0, "frequency", frequency)
    model_selection.insert(0, "asset", asset)
    model_selection["selected"] = model_selection["n_states"] == n_states
    return labels_df, summary, transitions, persistence, model_selection, feature_diagnostics

