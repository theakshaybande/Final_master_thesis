# Controlled LoCoMotif Run Report

## Data files used
- `final_dataset/features/crypto/BTCUSDT_15m_features_2020_2025.parquet`
- `final_dataset/features/crypto/BTCUSDT_1h_features_2020_2025.parquet`

## Regime labels
- agnostic_1h: agnostic from `not_applicable`
- high_vol: regenerated_for_controlled_experiment from `/home/i6404275/Final_master_thesis/reports/locomotif_controlled_slice_comparison/tables/btcusdt_15m_quantile2rolling240_regenerated_labels.csv`
- low_vol: regenerated_for_controlled_experiment from `/home/i6404275/Final_master_thesis/reports/locomotif_controlled_slice_comparison/tables/btcusdt_15m_quantile2rolling240_regenerated_labels.csv`

## Selected slices
- agnostic_1h: 2020-01-11T00:00:00+00:00 to 2020-01-23T11:00:00+00:00, n=300, rule=first_valid_contiguous_1h_window_after_warmup
- high_vol: 2020-12-16T14:00:00+00:00 to 2020-12-21T22:30:00+00:00, n=500, rule=longest_contiguous_high_vol_segment
- low_vol: 2025-06-25T07:45:00+00:00 to 2025-06-30T12:30:00+00:00, n=500, rule=longest_contiguous_low_vol_segment

## LoCoMotif package/import used
- Real `dtai-locomotif` through `locomotif.locomotif.apply_locomotif` when available.

## LoCoMotif parameter settings
- 15m: lmin=32, lmax=128, rho from CLI, nb=10, overlap=0.20, warping=True.
- 1h: lmin=24, lmax=72, rho from CLI, nb=10, overlap=0.20, warping=True.

## Matrix Profile parameter settings
- 15m windows: 32, 64, 128.
- 1h windows: 24, 48, 72.

## Success/failure status
- LoCoMotif agnostic_1h_close_z_rho0.65_n300: success=True, runtime=1.649134243838489, error=nan
- LoCoMotif low_vol_close_z_rho0.65_n500: success=True, runtime=1.5872423723340034, error=nan
- LoCoMotif high_vol_close_z_rho0.65_n500: success=True, runtime=1.6392561960965395, error=nan
- Matrix Profile agnostic_1h_close_z_m24_n300: best_distance=1.097756622181075, runtime=20.9546563597396
- Matrix Profile agnostic_1h_close_z_m48_n300: best_distance=2.741402888536006, runtime=0.2766603482887149
- Matrix Profile agnostic_1h_close_z_m72_n300: best_distance=3.4051174838788425, runtime=0.0011152476072311
- Matrix Profile low_vol_close_z_m32_n500: best_distance=1.4516092467236272, runtime=19.47117221262306
- Matrix Profile low_vol_close_z_m64_n500: best_distance=2.592858384704644, runtime=0.0024879146367311
- Matrix Profile low_vol_close_z_m128_n500: best_distance=7.683076291314106, runtime=0.002139450982213
- Matrix Profile high_vol_close_z_m32_n500: best_distance=1.5467674927294324, runtime=20.00468649808317
- Matrix Profile high_vol_close_z_m64_n500: best_distance=2.7162020324978604, runtime=0.0019240472465753
- Matrix Profile high_vol_close_z_m128_n500: best_distance=6.915562270853985, runtime=0.0017439154908061

## Generated tables
- `tables/locomotif_controlled_runtime.csv`
- `tables/locomotif_controlled_motif_sets.csv`
- `tables/locomotif_controlled_occurrences.csv`
- `tables/mp_controlled_slice_runtime.csv`
- `tables/mp_controlled_slice_motifs.csv`
- `tables/mp_vs_locomotif_controlled_comparison.csv`

## Generated figures
- See `figures/*.png`.

## Key numbers for thesis
- agnostic_1h LoCoMotif lmin=12, lmax=48, rho=0.65, nb=3, overlap=0.2, warping=True: count=3, runtime=1.649134243838489
- low_vol LoCoMotif lmin=12, lmax=48, rho=0.65, nb=3, overlap=0.2, warping=True: count=3, runtime=1.5872423723340034
- high_vol LoCoMotif lmin=12, lmax=48, rho=0.65, nb=3, overlap=0.2, warping=True: count=3, runtime=1.6392561960965395
- agnostic_1h Matrix Profile window_length=24: count=1, runtime=20.9546563597396
- agnostic_1h Matrix Profile window_length=48: count=1, runtime=0.2766603482887149
- agnostic_1h Matrix Profile window_length=72: count=1, runtime=0.0011152476072311
- low_vol Matrix Profile window_length=32: count=1, runtime=19.47117221262306
- low_vol Matrix Profile window_length=64: count=1, runtime=0.0024879146367311
- low_vol Matrix Profile window_length=128: count=1, runtime=0.002139450982213
- high_vol Matrix Profile window_length=32: count=1, runtime=20.00468649808317
- high_vol Matrix Profile window_length=64: count=1, runtime=0.0019240472465753
- high_vol Matrix Profile window_length=128: count=1, runtime=0.0017439154908061

## Limitations
- Controlled BTCUSDT slices only; not a full benchmark.
- MP and LoCoMotif output different object types, so raw counts are not equivalent.
- LoCoMotif runtime depends on slice length, rho, and motif length bounds.

## Exact commands to reproduce
```powershell
python scripts/run_locomotif_controlled_slice_comparison.py --mode smoke --slice agnostic_1h --feature close_z --max-points 1000 --rho 0.65 --run-mp yes --run-locomotif yes
python scripts/run_locomotif_controlled_slice_comparison.py --mode full --slice high_vol --feature close_z --max-points 2000 --rho 0.65 --run-mp yes --run-locomotif yes
python scripts/run_locomotif_controlled_slice_comparison.py --mode full --slice low_vol --feature close_z --max-points 2000 --rho 0.65 --run-mp yes --run-locomotif yes
```