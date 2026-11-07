"""Verify completed-run exports and rebuild Appendix C without model calls.

Uses the standard library. Historical RQ1 inputs are read-only; the two outputs
are the new appendix table and its deterministic evidence checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/concise-definition-supplement"
MODELS = {"llama3_8b": "Llama", "qwen2_7b": "Qwen2", "granite3_2_8b": "Granite"}
VIEWS = {"T_All": r"$T_{\mathrm{All}}$", "T_NFR": r"$T_{\mathrm{NFR}}$"}
TOL = 1e-12


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert abs(float(a) - float(b)) <= TOL, (a, b)


def main():
    score_path = DATA / "analysis/concise_scores.csv"
    group_path = DATA / "analysis/optimizer_group_comparisons.csv"
    pair_path = DATA / "analysis/paired_comparisons.csv"
    original_path = ROOT / "data/rq1/target_cells.csv"
    scores, groups, pairs, originals = map(rows, [score_path, group_path, pair_path, original_path])
    # RQ1 also exports two label-word-free sensitivity views. Appendix C uses
    # only the two prespecified full/NFR primary masks, without changing others.
    originals = [r for r in originals if r["view"] in VIEWS]
    config = json.loads((DATA / "frozen_config.json").read_text())
    audit = json.loads((DATA / "completion_audit.json").read_text())
    summary = json.loads((DATA / "summary.json").read_text())
    assert audit["status"] == "PASS" and summary["status"] == "complete"
    assert summary["routes"] == 3 and summary["logical_votes"] == 56718
    assert config["spec"]["votes_per_item"] == 3
    assert len(scores) == 6 and len(groups) == 12 and len(pairs) == 66 and len(originals) == 60
    score_map = {(r["model"], r["view"]): r for r in scores}
    group_map = {(r["model"], r["optimizer"], r["view"]): r for r in groups}
    assert len(score_map) == 6 and len(group_map) == 12
    definition = ROOT / "research/concise_definition_candidate.txt"
    definition_bytes = definition.read_bytes()
    assert definition_bytes.decode("utf-8") == config["definition_text"]
    assert hashlib.sha256(definition_bytes).hexdigest() == config["spec"]["definition_sha256"]
    table = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"View & Model & $p_0$ & Concise & PE2 mean & BT-APE mean \\", r"\midrule"]
    evidence_rows = []
    for vi, (view, display_view) in enumerate(VIEWS.items()):
        if vi:
            table.append(r"\midrule")
        for model, name in MODELS.items():
            concise = score_map[model, view]
            tp, tn, fp, fn = (int(concise[k]) for k in ("tp", "tn", "fp", "fn"))
            assert tp + tn + fp + fn == int(concise["n"]) == (6302 if view == "T_All" else 5771)
            rebuilt = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
            close(rebuilt, concise["macro_f1"])
            logged = [r for r in summary["scores"] if r["model"] == model and r["view"] == view]
            assert len(logged) == 1
            close(logged[0]["macro_f1"], concise["macro_f1"])
            base = [r for r in pairs if r["model"] == model and r["view"] == view and r["reference_prompt"] == "p0"]
            assert len(base) == 1
            p0 = float(base[0]["reference_macro_f1"])
            means = []
            for optimizer in ("pe2", "btape"):
                group = group_map[model, optimizer, view]
                history = [r for r in originals if (r["model"], r["optimizer"], r["view"]) == (model, optimizer, view)]
                assert len(history) == 5 and {int(r["seed"]) for r in history} == set(range(40, 45))
                assert int(group["historical_source_runs"]) == 5
                for original in history:
                    close(p0, original["p0_macro_f1"])
                    paired = [r for r in pairs if r["reference_prompt"] == original["cell_id"] and r["view"] == view]
                    assert len(paired) == 1
                    close(paired[0]["reference_macro_f1"], original["pstar_macro_f1"])
                    close(paired[0]["concise_macro_f1"], concise["macro_f1"])
                endpoint_mean = mean(float(r["pstar_macro_f1"]) for r in history)
                close(endpoint_mean, group["reference_macro_f1_mean"])
                close(concise["macro_f1"], group["concise_macro_f1_mean"])
                close(100 * (float(concise["macro_f1"]) - endpoint_mean), group["concise_minus_reference_macro_f1_pp_mean"])
                means.append(endpoint_mean)
            values = [p0, float(concise["macro_f1"]), *means]
            table.append(" & ".join([display_view, name, *(f"{100*x:.2f}" for x in values)]) + r" \\")
            evidence_rows.append(dict(model=model, view=view, p0=values[0], concise=values[1], pe2_mean=values[2], btape_mean=values[3]))
    table += [r"\bottomrule", r"\end{tabular}"]
    output = ROOT / "tables/concise_definition_comparison.tex"
    output.write_text("\n".join(table) + "\n", encoding="utf-8")
    checks = {"status": "PASS", "run_id": audit["run_id"], "source_commit": audit["commit"],
              "tolerance": TOL, "table_rows": evidence_rows, "historical_endpoint_checks": 60,
              "new_model_calls": 0, "historical_inputs_modified": False,
              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (score_path, group_path, pair_path, original_path, definition,
                                         DATA / "frozen_config.json", DATA / "completion_audit.json", DATA / "summary.json")},
              "table_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT / "research/concise_definition_asset_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
