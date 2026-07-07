# HPC Results Summary

## Jobs Found
- `logs/locomotif_high_tiny_6046.out`: bytes=1001, complete=True, timeout=False, error=False
- `logs/locomotif_low_tiny_6047.out`: bytes=1000, complete=True, timeout=False, error=False
- `logs/locomotif_micro_6045.out`: bytes=997, complete=True, timeout=False, error=False
- `logs/locomotif_high_tiny_6046.err`: bytes=0, complete=False, timeout=False, error=False
- `logs/locomotif_low_tiny_6047.err`: bytes=0, complete=False, timeout=False, error=False
- `logs/locomotif_micro_6045.err`: bytes=0, complete=False, timeout=False, error=False

## LoCoMotif Runtime and Motif Sets
- Runtime table: `reports/locomotif_controlled_slice_comparison/tables/locomotif_controlled_runtime.csv`
- agnostic_1h_close_z_rho0.65_n300: success, runtime=1.649134243838489, motif_sets=3, occurrences=18, error=nan
- low_vol_close_z_rho0.65_n500: success, runtime=1.5872423723340034, motif_sets=3, occurrences=28, error=nan
- high_vol_close_z_rho0.65_n500: success, runtime=1.6392561960965395, motif_sets=3, occurrences=27, error=nan

## Matrix Profile Results
- Matrix Profile table: `reports/locomotif_controlled_slice_comparison/tables/mp_controlled_slice_motifs.csv`
- agnostic_1h m=24: best_distance=1.097756622181075, runtime=20.9546563597396
- agnostic_1h m=48: best_distance=2.741402888536006, runtime=0.2766603482887149
- agnostic_1h m=72: best_distance=3.4051174838788425, runtime=0.0011152476072311
- low_vol m=32: best_distance=1.4516092467236272, runtime=19.47117221262306
- low_vol m=64: best_distance=2.592858384704644, runtime=0.0024879146367311
- low_vol m=128: best_distance=7.683076291314106, runtime=0.002139450982213
- high_vol m=32: best_distance=1.5467674927294324, runtime=20.00468649808317
- high_vol m=64: best_distance=2.7162020324978604, runtime=0.0019240472465753
- high_vol m=128: best_distance=6.915562270853985, runtime=0.0017439154908061

## Configs
- `reports/locomotif_controlled_slice_comparison/configs/locomotif_parameter_mapping.json`: keys=implementation, package, mapping
- `reports/locomotif_controlled_slice_comparison/configs/selected_slices_metadata.json`: 3 entries

## Generated Figures
- `reports/locomotif_controlled_slice_comparison/figures/btcusdt_15m_high_vol_slice_overview.png` (343126 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/btcusdt_15m_low_vol_slice_overview.png` (367171 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/controlled_locomotif_experiment_summary.png` (223176 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/locomotif_btcusdt_15m_high_vol_top_motif_set.png` (244210 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/locomotif_btcusdt_15m_low_vol_top_motif_set.png` (297567 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/locomotif_interval_length_distribution.png` (70174 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/locomotif_runtime_by_rho.png` (80924 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/micro_locomotif_btcusdt_1h_top_motif_set.png` (216479 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/micro_locomotif_runtime.png` (109336 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/micro_mp_vs_locomotif_btcusdt_1h_side_by_side.png` (303069 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_btcusdt_15m_high_vol_top_motif_overlay.png` (241191 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_btcusdt_15m_low_vol_top_motif_overlay.png` (291121 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_vs_locomotif_counts.png` (100726 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_vs_locomotif_high_vol_side_by_side.png` (315927 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_vs_locomotif_low_vol_side_by_side.png` (409930 bytes)
- `reports/locomotif_controlled_slice_comparison/figures/mp_vs_locomotif_runtime.png` (86520 bytes)

## Thesis Interpretation Recommendation
At least one real LoCoMotif run succeeded. Report successful motif-set counts and keep timeouts separate.