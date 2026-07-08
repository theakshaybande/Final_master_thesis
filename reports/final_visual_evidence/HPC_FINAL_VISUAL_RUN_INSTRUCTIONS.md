# Final Visual Evidence HPC Run Instructions

## Local
git add notebooks scripts slurm reports/final_visual_evidence
git commit -m "Add final visual evidence HPC workflow"
git push

## HPC
cd ~/Final_master_thesis
git pull
source .thesis-env/bin/activate
python -m py_compile scripts/run_final_motif_candlestick_event_context.py
python -m py_compile scripts/run_final_mp_locomotif_visual_comparison.py
python -m py_compile scripts/run_final_top_motif_gallery_and_clustering.py
python -m py_compile scripts/run_final_visual_evidence_summary.py

## Submit
sbatch slurm/run_final_mp_locomotif_visual_comparison.slurm
sbatch slurm/run_final_motif_candlestick_event_context.slurm
sbatch slurm/run_final_top_motif_gallery_and_clustering.slurm
sbatch slurm/run_final_visual_evidence_summary.slurm

## Check
squeue -u $USER
ls -lt logs | head
tail -f logs/final_visual_*.out

## After completion
python scripts/run_final_visual_evidence_summary.py
cat reports/final_visual_evidence/FINAL_VISUAL_EVIDENCE_REPORT.md

## Commit HPC outputs
git add reports/final_visual_evidence logs/final_visual_*.out logs/final_visual_*.err
git commit -m "Add final visual evidence outputs"
git push

## Local pull
git pull
