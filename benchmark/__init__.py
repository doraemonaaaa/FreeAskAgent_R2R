"""Subgoal-level instruction-following benchmark for R2R-CE (docs/subgoal_benchmark_design.md).

Modules
  common          paths, R2R-CE / dense-gt loading, trace loading, nDTW
  build_subgoals  FGR2R chunks -> per-episode subgoal boundaries (data/subgoals_<set>.json)
  build_variants  SWAP / DROP-k instruction variants as extra habitat splits
  metrics         SGCR, SGCR@k, ISens, PathAttrib, LocFail, prefix-keep, skip rate
  selfcheck       reference-path agent and shortest-path oracle, r_b calibration
"""
