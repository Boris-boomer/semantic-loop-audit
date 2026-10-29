# Data Sources

This repository does not redistribute raw trajectories. The three source datasets are:

1. **SWE-agent trajectories** — `nebius/SWE-agent-trajectories`
   https://huggingface.co/datasets/nebius/SWE-agent-trajectories
   Used for Station 1. Save the subset under `data/subset2000/` (HuggingFace `load_from_disk` format).

2. **OpenClawBench** — anonymous release
   https://anonymous.4open.science/r/openclaw_bench-4378/
   Used for Station 2. The scripts read `dataset/gold_iaa_300/trajectory_packets/` only (300 gold trajectories, not the full 31,264). Place the unzipped root at `data/openclaw_bench-4378/`.

3. **CUAErrorBench** — `CyT1ng/cua_debugger_traj`
   https://huggingface.co/datasets/CyT1ng/cua_debugger_traj
   Used for Station 3. The script loads directly from HuggingFace or a local parquet at `data/cua_debugger/data/train.parquet`.

Derived files included in this repo (`data/windows_v2.jsonl`, `data/rmt-samples.jsonl`,
`data/sheaf-samples.jsonl`, `data/semantic-labels.jsonl`, `data/blind_audit_pack.txt`,
`data/blind_audit_key.json`) contain no raw trajectory text and no personal information.