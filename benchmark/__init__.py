"""Subgoal-level instruction-following benchmark for R2R-CE (docs/subgoal_benchmark_design.md).

Foundation
  common          paths (data/<set>/{rule,llm}/...), R2R-CE / dense-gt loading,
                  trace loading, split writing, turn geometry, nDTW
  build_subgoals  FGR2R human chunks -> per-episode subgoal boundaries B_k
                  (data/<set>/subgoals.json) -- every variant and metric builds on this

Variants (data/<set>/rule/ -- minimal pairs, comparable to ORIG directly)
  build_variants  SWAP (donor instruction) and DROP-k (one sub-instruction removed)
  build_flip      FLIP-k, the main experiment: one left/right word reversed
  build_goalonly  GOAL-ONLY: only the last sub-instruction kept

Variants (data/<set>/llm/ -- LLM rewrites, only comparable to their own A1 control)
  build_paraphrase  PARAPHRASE arms para_id / para_terse / para_natural / para_lm_shift

Scoring and tooling
  metrics         SGCR(-eff), SGCR@k, ISens, PathAttrib, LocFail, Skip, PrefixKeep,
                  plus the FLIP and GOAL-ONLY evaluators; bootstrap 95% CI
  selfcheck       reference-path agent and shortest-path oracle, r_b calibration
  compact_trace   shrink a runner output dir to the fields the metrics read
  canav, awarevln adapters: build each system's inputs, import its trajectories
"""
