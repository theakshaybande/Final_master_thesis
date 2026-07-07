# Controlled LoCoMotif Run Report

## Data files used
- `final_dataset/features/crypto/BTCUSDT_15m_features_2020_2025.parquet`
- `final_dataset/features/crypto/BTCUSDT_1h_features_2020_2025.parquet`

## Regime labels
- agnostic_1h: agnostic from `not_applicable`
- high_vol: regenerated_for_controlled_experiment from `C:\Users\learn\OneDrive\Desktop\Final Masters Thesis\reports\locomotif_controlled_slice_comparison\tables\btcusdt_15m_quantile2rolling240_regenerated_labels.csv`
- low_vol: regenerated_for_controlled_experiment from `C:\Users\learn\OneDrive\Desktop\Final Masters Thesis\reports\locomotif_controlled_slice_comparison\tables\btcusdt_15m_quantile2rolling240_regenerated_labels.csv`

## Selected slices
- agnostic_1h: 2020-01-11T00:00:00+00:00 to 2020-01-17T05:00:00+00:00, n=150, rule=first_valid_contiguous_1h_window_after_warmup
- high_vol: 2020-12-16T14:00:00+00:00 to 2021-01-06T14:30:00+00:00, n=2000, rule=longest_contiguous_high_vol_segment
- low_vol: 2025-06-25T07:45:00+00:00 to 2025-07-16T03:30:00+00:00, n=2000, rule=longest_contiguous_low_vol_segment

## LoCoMotif package/import used
- Real `dtai-locomotif` through `locomotif.locomotif.apply_locomotif` when available.

## LoCoMotif parameter settings
- 15m: lmin=32, lmax=128, rho from CLI, nb=10, overlap=0.20, warping=True.
- 1h: lmin=24, lmax=72, rho from CLI, nb=10, overlap=0.20, warping=True.

## Matrix Profile parameter settings
- 15m windows: 32, 64, 128.
- 1h windows: 24, 48, 72.

## Success/failure status
- LoCoMotif agnostic_1h_close_z_rho0.65_n1000: success=False, runtime=nan, error=TimeoutError('Real LoCoMotif call exceeded 360 seconds.')
- LoCoMotif high_vol_close_z_rho0.65_n2000: success=False, runtime=nan, error=TimeoutError('Real LoCoMotif call exceeded 120 seconds.')
- LoCoMotif low_vol_close_z_rho0.65_n2000: success=False, runtime=nan, error=TimeoutError('Real LoCoMotif call exceeded 120 seconds.')
- LoCoMotif agnostic_1h_close_z_rho0.65_n300: success=False, runtime=nan, error=TimeoutError('Real LoCoMotif call exceeded 600 seconds.')
- LoCoMotif agnostic_1h_close_z_rho0.65_n150: success=False, runtime=nan, error=TimeoutError('Real LoCoMotif call exceeded 600 seconds.')
- Matrix Profile agnostic_1h_close_z_m24_n1000: best_distance=0.864456494113353, runtime=25.25699860000168
- Matrix Profile agnostic_1h_close_z_m48_n1000: best_distance=1.6881810735680085, runtime=0.0272111000012955
- Matrix Profile agnostic_1h_close_z_m72_n1000: best_distance=2.157387399990955, runtime=0.0223655000008875
- Matrix Profile high_vol_close_z_m32_n2000: best_distance=1.2596553869820644, runtime=24.8668061999997
- Matrix Profile high_vol_close_z_m64_n2000: best_distance=1.777806546219947, runtime=0.0311748000021907
- Matrix Profile high_vol_close_z_m128_n2000: best_distance=3.206065176956419, runtime=0.0298269000049913
- Matrix Profile low_vol_close_z_m32_n2000: best_distance=1.0140356654521008, runtime=25.29417839999951
- Matrix Profile low_vol_close_z_m64_n2000: best_distance=1.862371282034455, runtime=0.0282050000023446
- Matrix Profile low_vol_close_z_m128_n2000: best_distance=2.5783948046574103, runtime=0.0284736000030534
- Matrix Profile agnostic_1h_close_z_m24_n300: best_distance=1.097756622181075, runtime=29.13620819999778
- Matrix Profile agnostic_1h_close_z_m48_n300: best_distance=2.741402888536006, runtime=0.0051800000001094
- Matrix Profile agnostic_1h_close_z_m72_n300: best_distance=3.4051174838788425, runtime=0.0058575000002747
- Matrix Profile agnostic_1h_close_z_m24_n150: best_distance=1.0977566221810604, runtime=33.03948400000081
- Matrix Profile agnostic_1h_close_z_m48_n150: best_distance=4.032606788574689, runtime=0.0054463999986182
- Matrix Profile agnostic_1h_close_z_m72_n150: best_distance=6.2136848150774, runtime=0.0061176000017439

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
- agnostic_1h LoCoMotif lmin=24, lmax=72, rho=0.65, nb=10, overlap=0.2, warping=True: count=0, runtime=nan
- high_vol LoCoMotif lmin=32, lmax=128, rho=0.65, nb=10, overlap=0.2, warping=True: count=0, runtime=nan
- low_vol LoCoMotif lmin=32, lmax=128, rho=0.65, nb=10, overlap=0.2, warping=True: count=0, runtime=nan
- agnostic_1h LoCoMotif lmin=12, lmax=48, rho=0.65, nb=3, overlap=0.2, warping=True: count=0, runtime=nan
- agnostic_1h LoCoMotif lmin=8, lmax=32, rho=0.65, nb=2, overlap=0.2, warping=True: count=0, runtime=nan
- agnostic_1h Matrix Profile window_length=24: count=1, runtime=25.25699860000168
- agnostic_1h Matrix Profile window_length=48: count=1, runtime=0.0272111000012955
- agnostic_1h Matrix Profile window_length=72: count=1, runtime=0.0223655000008875
- high_vol Matrix Profile window_length=32: count=1, runtime=24.8668061999997
- high_vol Matrix Profile window_length=64: count=1, runtime=0.0311748000021907
- high_vol Matrix Profile window_length=128: count=1, runtime=0.0298269000049913
- low_vol Matrix Profile window_length=32: count=1, runtime=25.29417839999951
- low_vol Matrix Profile window_length=64: count=1, runtime=0.0282050000023446
- low_vol Matrix Profile window_length=128: count=1, runtime=0.0284736000030534
- agnostic_1h Matrix Profile window_length=24: count=1, runtime=29.13620819999778
- agnostic_1h Matrix Profile window_length=48: count=1, runtime=0.0051800000001094
- agnostic_1h Matrix Profile window_length=72: count=1, runtime=0.0058575000002747
- agnostic_1h Matrix Profile window_length=24: count=1, runtime=33.03948400000081
- agnostic_1h Matrix Profile window_length=48: count=1, runtime=0.0054463999986182
- agnostic_1h Matrix Profile window_length=72: count=1, runtime=0.0061176000017439

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