# Semantic Loop Audit — Companion Repository

Code and derived data for:

> Semantic Loops in LLM Agent Trajectories: A Construct-Validity Audit of Three Public Resources

## Contents

- `code/` — seven scripts, one per analysis in the paper
- `data/` — derived files needed to reproduce Tables 1, 1a, 2 (no raw trajectories)
- `DATA_SOURCES.md` — where the three raw datasets come from

## Reproduce Table 1a and inter-annotator kappa (no external data needed)

    uv run python code/compare_annotators.py

Output should match the numbers reported in the paper (3-class kappa = 0.214,
2-class kappa = 0.500, Cell-00: 3 YES / 2 MAYBE / 2 NO).

## Reproduce Table 3 (OpenClawBench)

Download the OpenClawBench gold set first (see `DATA_SOURCES.md`), place it under
`data/openclaw_bench-4378/`, then:

    uv run --with pandas python code/probe_openclaw_loop.py

## Reproduce R4 census (CUAErrorBench)

Loads from HuggingFace (mirror supported via `HF_ENDPOINT`):

    uv run --with datasets --with pyarrow python code/probe_cua_debugger.py

## Reproduce Station 1 window construction (requires SWE-agent subset2000)

Download `nebius/SWE-agent-trajectories` into `data/subset2000/` (see `DATA_SOURCES.md`),
then run in order:

    uv run --with datasets python code/build_windows_v2.py
    uv run --with datasets --with scikit-learn --with scipy python code/eval_semantic_label.py
    uv run --with datasets python code/probe_four_cell_audit.py
    uv run --with datasets python code/build_blind_audit_pack.py
    uv run python code/compare_annotators.py

If the data directory is not at `../data` relative to `code/`, set:

    $env:TRAJGAP_DATA = "path/to/your/data"