"""Plot frozen source-test sensitivity results in the manuscript's existing style.

Run: uv run --isolated --no-project --with matplotlib --with numpy python build_figure.py
All data come from the parent supplement; no inference or interval estimation.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import orx_figstyle as style

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
WIDTH = 372 / 72.27
MODELS = ["llama3_8b", "qwen2_7b", "granite3_2_8b"]
NAMES = ["Llama-3", "Qwen2", "Granite-3.2"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "^", "s"]
VIEWS = ["original_rows", "unseen_rows", "unseen_equal_text_weights"]
LABELS = ["Original\nrows", "Overlap\nremoved", "Equal text\nweights"]


def main():
    with (DATA / "cell-test-sensitivity.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    index = {(r["cell_id"], r["source_test_view"]): r for r in rows}
    originals = [r for r in rows if r["source_test_view"] == VIEWS[0]]
    assert len(originals) == 30 and len(rows) == 90
    summary = json.loads((DATA / "audit-results.json").read_text())
    style.TEXT = WIDTH
    style.use_style()
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                        "xtick.labelsize": 8, "ytick.labelsize": 8,
                        "legend.fontsize": 8, "savefig.dpi": 300})
    wp, height = WIDTH * 72, 270
    fig = plt.figure(figsize=(WIDTH, height / 72))
    axes = [fig.add_axes([x / wp, 47 / height, 132 / wp, 144 / height])
            for x in [44, 225]]
    plotted = []
    for panel, ax in enumerate(axes):
        views = VIEWS[1:] if panel == 0 else VIEWS
        for x, view in enumerate(views):
            vals = []
            for row in originals:
                current = index[row["cell_id"], view]
                value = (float(current["pstar_macro_f1_pp"]) - float(row["pstar_macro_f1_pp"])
                         if panel == 0 else float(current["selection_gap_pp"]))
                vals.append(value)
                mi = MODELS.index(row["model"])
                oi = ["pe2", "btape"].index(row["optimizer"])
                offset = (mi - 1) * .17 + (int(row["seed"]) - 42) * .024 + (oi - .5) * .014
                ax.scatter(x + offset, value, s=14, marker=MARKERS[mi],
                           facecolor=COLORS[mi] if oi == 0 else "white",
                           edgecolor=COLORS[mi], linewidth=.7, alpha=.85, zorder=3)
                plotted.append({"panel": "a" if panel == 0 else "b",
                                "cell_id": row["cell_id"], "model": row["model"],
                                "optimizer": row["optimizer"], "seed": row["seed"],
                                "view": view, "value_pp": value})
            value = float(np.mean(vals) if panel == 0 else np.median(vals))
            expected = summary["source_test_sensitivity"][view][
                "pstar_macro_f1_pp_change_from_original" if panel == 0 else "selection_gap_pp"
            ]["mean" if panel == 0 else "median"]
            assert abs(value - expected) < 1e-12
            ax.plot([x - .31, x + .31], [value, value], color="#222222", lw=1, zorder=4)
            ax.scatter([x], [value], marker="D", s=22, color="#222222",
                       edgecolor="white", linewidth=.6, zorder=5)
            ax.text(x, 1.035, f"{value:+.3f}" if panel == 0 else f"{value:.3f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(views)), LABELS[1:] if panel == 0 else LABELS)
        ax.set_xlim(-.5, len(views) - .5)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_xlabel("Source-test scoring", labelpad=6)
        ax.set_facecolor("#FAFBFC")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True, color="#E3E8EC", lw=.4)
        ax.text(0, 1.18, "(a) Macro-F1 change" if panel == 0 else "(b) Selection gap",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, fontweight="bold", color="#245C78")
    axes[0].set_ylabel("Change from original (pp)")
    axes[0].set_ylim(-4.3, 2)
    axes[0].set_yticks([-4, -3, -2, -1, 0, 1, 2])
    axes[0].axhline(0, color="#707A81", ls="--", lw=.8, zorder=1)
    axes[1].set_ylabel("Macro-F1 gap (pp)")
    axes[1].set_ylim(-.2, 4.6)
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].axhline(1, color="#707A81", ls="--", lw=.8, zorder=1)
    # Model colours and source-optimizer names match the existing figures.
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls="", label=n)
                                 for c, m, n in zip(COLORS, MARKERS, NAMES)],
                         loc="upper center", bbox_to_anchor=(.53, .995), ncol=3,
                         handletextpad=.5, columnspacing=1.3)
    fig.legend(handles=[Line2D([], [], color="#555555", marker="o", ls="", label="Filled: PE2"),
                        Line2D([], [], mec="#555555", mfc="white", marker="o", ls="", label="Open: BT-APE"),
                        Line2D([], [], color="#222222", marker="D", lw=1,
                               label="Mean (a); median (b)")],
               loc="upper center", bbox_to_anchor=(.53, .925), ncol=3,
               handletextpad=.5, columnspacing=1.1)
    # The observed ranges must fit; this check guards against silently clipping data.
    for panel, ax in zip(["a", "b"], axes):
        values = [r["value_pp"] for r in plotted if r["panel"] == panel]
        assert min(values) > ax.get_ylim()[0] and max(values) < ax.get_ylim()[1]
    issues = style._audit(fig)
    report = {"issues": issues, "printed_width_inches": WIDTH,
              "matplotlib_version": mpl.__version__, "numpy_version": np.__version__,
              "font": style._resolved_font, "plotted_cell_values": len(plotted),
              "summary_matches_audit": True,
              "input_sha256": {n: hashlib.sha256((DATA / n).read_bytes()).hexdigest()
                               for n in ["cell-test-sensitivity.csv", "audit-results.json"]}}
    (HERE / "figure-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if issues:
        fig.savefig(HERE / "layout-review.png", dpi=180)
        raise AssertionError(issues)
    style.save(fig, str(HERE / "Fig11-source-duplicates"), formats=("pdf", "svg", "png"))
    with (HERE / "plotted-values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
