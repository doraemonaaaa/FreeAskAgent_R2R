"""Every benchmark result as English markdown tables (results/TABLES.md).

    python -m benchmark.tables [--out benchmark/results/TABLES.md]

One table per experiment, every cell filled from the archived traces:
ORIG, run-to-run / cross-system agreement, FLIP, GOAL-ONLY, PARAPHRASE on the
200-set (AwareVLN, CA-Nav) and on the 100-episode set shared by all three
systems. OSR is read from the ``osr=`` field of the result lines; RSR = SR / OSR.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from .common import ROOT, data_path, load_gt, load_json, load_run, ndtw, path_walked, read_id_list
from .metrics import evaluate_flip, evaluate_paired, score_episode

R = ROOT / "benchmark/results"
OSR = re.compile(r" id=(\S+) steps=\d+ .*? osr=([\d.]+)")
ARMS = ("orig", "para_id", "para_terse", "para_natural", "para_lm_shift")
ARM_NAME = {"orig": "ORIG", "para_id": "A1 same-register", "para_terse": "A2 terse",
            "para_natural": "A3 colloquial", "para_lm_shift": "A4 landmark shift"}
SHORT = {"orig": "ORIG", "para_id": "A1", "para_terse": "A2", "para_natural": "A3", "para_lm_shift": "A4"}
CONTROL = {"para_id": "orig", "para_terse": "para_id", "para_natural": "para_id", "para_lm_shift": "para_id"}

SG = load_json(data_path("subgoals"))["episodes"]
GT = load_gt("val_unseen")
rng = np.random.default_rng(0)
_cache = {}


def run(*dirs):
    key = tuple(str(d) for d in dirs)
    if key not in _cache:
        positions, results = load_run(list(key))
        for d in key:
            for log in Path(d).glob("rank_*.log"):
                for m in OSR.finditer(log.read_text(errors="replace")):
                    if m.group(1) in results:
                        results[m.group(1)]["osr"] = float(m.group(2))
        _cache[key] = (positions, results)
    return _cache[key]


def ci(values, bold=True):
    v = np.asarray(values, dtype=np.float64)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    text = "{:+.3f} [{:+.2f}, {:+.2f}]".format(v.mean(), lo, hi)
    return "**{}**".format(text) if bold and (lo > 0 or hi < 0) else text


def sig(value, p):
    return "**{}**".format(value) if p < 0.05 else value


def per_run(r, eps):
    pos, res = r
    sr = np.mean([res[e]["success"] > 0 for e in eps])
    osr = np.mean([res[e].get("osr", float("nan")) > 0 for e in eps])
    return dict(n=len(eps), sr=sr, osr=osr, rsr=sr / osr if osr else float("nan"),
                spl=np.mean([res[e]["spl"] for e in eps]), ne=np.mean([res[e]["dtg"] for e in eps]),
                ndtw=np.mean([ndtw(pos[e], GT[e]) for e in eps]),
                sgcr=np.mean([score_episode(pos[e], SG[e]["subgoals"], 1.5)["c"] / SG[e]["K"] for e in eps if e in SG]),
                pl=np.mean([path_walked(pos[e])[-1] for e in eps]), steps=np.mean([res[e]["steps"] for e in eps]))


def paired(a, b, eps):
    m, _ = evaluate_paired(a, b, ids=set(eps))
    return m


def table_orig():
    rows = [("FreeAskAgent (2026-09-19 code)", run(R / "freeaskagent/orig")), ("CA-Nav (GPT-4 parse)", run(R / "canav/orig")),
            ("CA-Nav (Qwen parse)", run(R / "canav_qwen/orig")), ("AwareVLN", run(R / "awarevln/orig"))]
    out = ["### Table 1. ORIG, val_unseen_200 (↑ higher is better, ↓ lower is better)", "",
           "| System | n | SR ↑ | OSR ↑ | RSR ↑ | SPL ↑ | NE (m) ↓ | nDTW ↑ | SGCR ↑ | Path length (m) | Steps |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        eps = sorted(r[1], key=int)
        s = per_run(r, eps)
        out.append("| {} | {n} | {sr:.3f} | {osr:.3f} | {rsr:.3f} | {spl:.3f} | {ne:.2f} | {ndtw:.3f} | {sgcr:.3f} | {pl:.1f} | {steps:.0f} |".format(name, **s))
    return out


def table_agreement():
    pairs = [("FreeAskAgent ORIG vs ORIG rerun (noise floor)", run(R / "freeaskagent/orig"), run(R / "freeaskagent/orig2")),
             ("CA-Nav GPT-4 parse vs Qwen parse", run(R / "canav/orig"), run(R / "canav_qwen/orig")),
             ("FreeAskAgent vs CA-Nav (ORIG)", run(R / "freeaskagent/orig"), run(R / "canav/orig")),
             ("FreeAskAgent vs AwareVLN (ORIG)", run(R / "freeaskagent/orig"), run(R / "awarevln/orig")),
             ("CA-Nav vs AwareVLN (ORIG)", run(R / "canav/orig"), run(R / "awarevln/orig"))]
    out = ["### Table 2. Run-to-run and cross-system agreement on success", "",
           "| Pair | n | SR (A → B) | Solved by both / A only / B only | Flip rate | κ ↑ | McNemar p |",
           "|---|---|---|---|---|---|---|"]
    for name, a, b in pairs:
        eps = sorted(set(a[1]) & set(b[1]), key=int)
        m = paired(a, b, eps)
        out.append("| {} | {} | {:.3f} → {:.3f} | {:.0f} / {:.0f} / {:.0f} | {:.3f} | {:.2f} | {:.2f} |".format(
            name, len(eps), m["SR_orig"]["mean"], m["SR_other"]["mean"], m["Solved_both"]["mean"], m["Solved_orig_only"]["mean"],
            m["Solved_other_only"]["mean"], m["FlipRate"]["mean"], m["Kappa"]["mean"], m["McNemar_p"]["mean"]))
    return out


def table_flip():
    meta = load_json(data_path("flip", "val_unseen"))
    out = ["### Table 3. FLIP: turn word of the first sub-instruction reversed (82 episodes, v1-3 start heading)", "",
           "Per run, each episode is one of follow / opposite / straight / none (rates sum to 1). "
           "WordEffect = MeanFollow − Blind: 0 = the word is ignored, ≈ +0.5 = fully obedient.", "",
           "| System | Run | Follow ↑ | Opposite ↓ | Straight | None | SR ↑ |", "|---|---|---|---|---|---|---|"]
    summary = ["", "| System | MeanFollow ↑ | Blind | WordEffect ↑ [95% CI] | Obeyed in both runs ↑ | Same side in both runs ↓ | Follow given *left* | Follow given *right* | κ (success) | McNemar p |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in (("FreeAskAgent", "freeaskagent"), ("AwareVLN", "awarevln"), ("CA-Nav", "canav")):
        o, f = run(R / d / "flip_orig"), run(R / d / "flip")
        m, per = evaluate_flip(o, f, meta)
        for tag, label in (("ORIG", "ORIG (original word)"), ("FLIP", "FLIP (reversed word)")):
            out.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                name, label, m[tag + "_follow"]["mean"], m[tag + "_opposite"]["mean"], m[tag + "_straight"]["mean"],
                m[tag + "_none"]["mean"], m["SR_" + tag.lower()]["mean"]))
        p = paired(o, f, list(per))
        we = m["WordEffect"]
        summary.append("| {} | {:.3f} | {:.3f} | {:+.3f} [{:+.2f}, {:+.2f}] | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
            name, m["MeanFollow"]["mean"], m["Blind"]["mean"], we["mean"], we["ci"][0], we["ci"][1], m["BothFollow"]["mean"],
            m["SameSide"]["mean"], m["FollowGiven_left"]["mean"], m["FollowGiven_right"]["mean"], p["Kappa"]["mean"], p["McNemar_p"]["mean"]))
    return out + summary


def table_goalonly():
    goals = load_json(data_path("goalonly"))["goalonly"]
    out = ["### Table 4. GOAL-ONLY: only the last sub-instruction kept (val_unseen_200)", "",
           "Δ = GOAL-ONLY − ORIG. A large nDTW / intermediate-boundary drop means the agent was following the route description.", "",
           "| System | n | SR (ORIG → GOAL-ONLY) | κ | McNemar p | ΔnDTW [95% CI] | ΔPath length (m) [95% CI] | Intermediate boundaries (ORIG → GOAL-ONLY) | SR kept | SR gained |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in (("FreeAskAgent (2026-09-19 code)", "freeaskagent"), ("AwareVLN", "awarevln"), ("CA-Nav (GPT-4 parse)", "canav")):
        o, g = run(R / d / "orig"), run(R / d / "goalonly")
        eps = sorted((set(goals) & set(o[1]) & set(g[1]) & set(o[0]) & set(g[0])), key=int)
        m = paired(o, g, eps)

        def inter(r, e):
            ent = score_episode(r[0][e], SG[e]["subgoals"], 1.5)["entries"][:-1]
            return sum(x is not None for x in ent) / max(len(ent), 1)
        kept = [g[1][e]["success"] > 0 for e in eps if o[1][e]["success"] > 0]
        gained = [g[1][e]["success"] > 0 for e in eps if o[1][e]["success"] == 0]
        out.append("| {} | {} | {:.3f} → {:.3f} | {:.2f} | {} | {} | {} | {:.3f} → {:.3f} | {:.3f} | {:.3f} |".format(
            name, len(eps), m["SR_orig"]["mean"], m["SR_other"]["mean"], m["Kappa"]["mean"], sig("{:.3f}".format(m["McNemar_p"]["mean"]), m["McNemar_p"]["mean"]),
            ci([ndtw(g[0][e], GT[e]) - ndtw(o[0][e], GT[e]) for e in eps]),
            ci([path_walked(g[0][e])[-1] - path_walked(o[0][e])[-1] for e in eps]),
            np.mean([inter(o, e) for e in eps]), np.mean([inter(g, e) for e in eps]), np.mean(kept), np.mean(gained)))
    return out


def tables_paraphrase(title, systems, eps):
    out = ["### {} — per-arm metrics (n = {})".format(title, len(eps)), "",
           "| System | Arm | SR ↑ | OSR ↑ | RSR ↑ | SPL ↑ | NE (m) ↓ | nDTW ↑ |", "|---|---|---|---|---|---|---|---|"]
    for name, dirs in systems.items():
        for a in ARMS:
            s = per_run(run(*dirs[a]), eps)
            out.append("| {} | {} | {sr:.3f} | {osr:.3f} | {rsr:.3f} | {spl:.3f} | {ne:.2f} | {ndtw:.3f} |".format(name, ARM_NAME[a], **s))
    out += ["", "### {} — paired comparison (n = {})".format(title, len(eps)), "",
            "A1 vs ORIG; A2–A4 vs A1. Δ = perturbed − control; ≈ 0 means robust to the rewording. **Bold**: p < 0.05 or 95% CI excluding 0.", "",
            "| System | Comparison | SR (control → perturbed) | ΔSR (≈0) | κ ↑ | McNemar p | ΔnDTW (≈0) [95% CI] |",
            "|---|---|---|---|---|---|---|"]
    for name, dirs in systems.items():
        for a, c in CONTROL.items():
            A, B = run(*dirs[c]), run(*dirs[a])
            m = paired(A, B, eps)
            p = m["McNemar_p"]["mean"]
            out.append("| {} | {} vs {} | {:.3f} → {:.3f} | {} | {:.2f} | {} | {} |".format(
                name, SHORT[a], SHORT[c], m["SR_orig"]["mean"], m["SR_other"]["mean"],
                sig("{:+.3f}".format(m["SR_other"]["mean"] - m["SR_orig"]["mean"]), p), m["Kappa"]["mean"],
                sig("{:.2f}".format(p), p), ci([ndtw(B[0][e], GT[e]) - ndtw(A[0][e], GT[e]) for e in eps])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(R / "TABLES.md"))
    args = parser.parse_args()
    ids200 = sorted(read_id_list(ROOT / "integrations/v3/eval_sets/val_unseen_200.txt"), key=int)
    ids100 = sorted(set(read_id_list(ROOT / "outputs/para50_20260925/ids.txt"))
                    | set(read_id_list(ROOT / "outputs/para50b_20260926/ids.txt")), key=int)
    two = {"AwareVLN": {a: [R / "awarevln" / a] for a in ARMS}, "CA-Nav (Qwen parse)": {a: [R / "canav_qwen" / a] for a in ARMS}}
    three = {"FreeAskAgent": {a: [R / "freeaskagent" / ("p50_" + a), R / "freeaskagent" / ("p50b_" + a)] for a in ARMS}, **two}
    lines = ["# Benchmark results (all experiments)", "",
             "Generated by `python -m benchmark.tables` from the archived traces in `benchmark/results/`. "
             "RSR = SR / OSR. FreeAskAgent ORIG / GOAL-ONLY are 2026-09-19 code (v1-2 start heading); "
             "its FLIP and PARAPHRASE are 2026-09-25/26 code (v1-3 start heading).", ""]
    for block in (table_orig(), table_agreement(), table_flip(), table_goalonly(),
                  tables_paraphrase("Table 5. PARAPHRASE, val_unseen_200", two, ids200),
                  tables_paraphrase("Table 6. PARAPHRASE, 100-episode subset shared by all systems", three, ids100)):
        lines += block + [""]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
