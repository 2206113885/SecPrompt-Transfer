"""Descriptive RQ3 matrix from frozen CSVs; no estimation or inference.

uv run --isolated --no-project --with numpy==2.5.4 --with matplotlib==3.11.2 \
    python figures/build_rq3_matrix_overview.py

The existing forest plots retain all source-split points and interval families.
"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import orx_figstyle as style

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "rq3-systematic"
WIDTH = 372 / 72.27
MODELS = ["llama3_8b", "qwen2_7b", "granite3_2_8b", "falcon3_7b",
          "ministral_8b", "granite3_3_8b", "qwen3_8b"]
SHORT = ["L3", "Q2", "G32", "F3", "M", "G33", "Q3"]
FAMILIES = [(p, o) for p in MODELS[:3] for o in ("pe2", "btape")]

def rows(name):
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))

def main():
    groups = [r for r in rows("group_intervals.csv")
              if r["view"] == "T_All" and r["metric"] == "macro_f1"]
    bases = [r for r in rows("baseline_intervals.csv")
             if r["view"] == "T_All" and r["metric"] == "macro_f1"]
    by_group = {(r["producer"], r["optimizer"], r["deployer"]): r for r in groups}
    by_model = {r["deployer"]: r for r in bases}
    assert len(groups) == len(by_group) == 42
    assert len(bases) == len(by_model) == 7
    assert all(int(r["source_seeds"]) == 5 for r in groups)
    grid = np.array([[float(by_group[p, o, d]["mean_gain_pp"])
                      for d in MODELS] for p, o in FAMILIES])
    baseline = np.array([100 * float(by_model[d]["estimate"]) for d in MODELS])
    assert np.isfinite(grid).all() and np.isfinite(baseline).all()
    for p, o in FAMILIES:
        for d in MODELS:
            row = by_group[p, o, d]
            assert abs(float(row["baseline_score"]) - float(by_model[d]["estimate"])) < 1e-12
            assert abs(100 * (float(row["mean_score"]) - float(row["baseline_score"]))
                       - float(row["mean_gain_pp"])) < 1e-10
            assert (row["origin_diagonal"] == "True") == (p == d)
    same_origin = [(i, MODELS.index(p)) for i, (p, _) in enumerate(FAMILIES)]
    assert len(same_origin) == 6
    style.TEXT = WIDTH
    style.use_style()
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "savefig.dpi": 300})
    fig = plt.figure(figsize=(WIDTH, 3.82))
    ax = fig.add_axes([.235, .405, .745, .385])
    base_ax = fig.add_axes([.235, .815, .745, .075])
    bar_ax = fig.add_axes([.37, .267, .50, .030])
    # Derive symmetric limits from every cell; zero remains neutral.
    limit = float(np.max(np.abs(grid)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    # Positive gains use the manuscript's blue register; negatives are warm.
    cmap = mpl.colormaps[style.DIVERGING].reversed()
    mesh = ax.pcolormesh(np.arange(8) - .5, np.arange(7) - .5, grid,
                        cmap=cmap, norm=norm, shading="flat", rasterized=False,
                        edgecolors="white", linewidth=.65)
    ax.set_xlim(-.5, 6.5)
    ax.set_ylim(5.5, -.5)
    ax.set_xticks([])
    ax.set_xlabel("Deployment model", labelpad=8)
    ax.xaxis.set_label_coords(.5, 1.46)
    ax.set_yticks(range(6), [SHORT[MODELS.index(p)] + " / " +
                           ("PE2" if o == "pe2" else "BT-APE") for p, o in FAMILIES])
    ax.set_ylabel("Prompt producer / optimizer", labelpad=8)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(6):
        for j in range(7):
            red, green, blue, _ = cmap(norm(grid[i, j]))
            luminance = .2126 * red + .7152 * green + .0722 * blue
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center",
                    color="white" if luminance < .45 else "black", fontsize=8)
    for i, j in same_origin:
        ax.add_patch(Rectangle((j-.46, i-.45), .92, .90, fill=False,
                               linewidth=1.1, edgecolor="#222222"))
    # Absolute baseline percentages use a neutral strip outside the gain scale.
    base_ax.set_xlim(-.5, 6.5)
    base_ax.set_ylim(0, 1)
    base_ax.set_axis_off()
    base_ax.add_patch(Rectangle((-.5, 0), 7, .53, facecolor="#F1F1F1", edgecolor="none"))
    for j, name in enumerate(SHORT):
        base_ax.text(j, .88, name, ha="center", va="center", fontweight="bold", fontsize=8)
        base_ax.text(j, .26, f"{baseline[j]:.2f}", ha="center", va="center", fontsize=8)
    base_ax.text(-.68, .26, r"$p_0$ (%)", ha="right", va="center", fontsize=8)
    colorbar = fig.colorbar(mesh, cax=bar_ax, orientation="horizontal")
    # Matplotlib otherwise rasterizes a dense colorbar even when the matrix
    # itself is vector; keep both parts editable and sharp in the PDF.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        # Avoid white hairline seams between vector colourbar patches in PDF viewers.
        colorbar.solids.set_edgecolor("face")
    colorbar.set_label("Mean macro-F1 gain over deployer $p_0$ (pp)", labelpad=3)
    colorbar.set_ticks([-20, -10, 0, 10, 20])
    colorbar.outline.set_visible(False)
    fig.text(.235, .365, "Outlined: same-origin reference (6); other cells: model switch (36)",
             ha="left", va="center", fontsize=8)
    fig.text(.235, .130, "L3: Llama-3-8B; Q2: Qwen2-7B; G32: Granite-3.2-8B", fontsize=8)
    fig.text(.235, .084, "F3: Falcon3-7B; M: Ministral-8B", fontsize=8)
    fig.text(.235, .038, "G33: Granite-3.3-8B; Q3: Qwen3-8B", fontsize=8)
    issues = style._audit(fig)
    audit = {
        "figure": "rq3_matrix_overview", "view": "T_All", "metric": "macro_f1",
        "input_sha256": {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                         for name in ("group_intervals.csv", "baseline_intervals.csv")},
        "group_count": len(groups), "same_origin_count": len(same_origin),
        "model_switch_count": len(groups) - len(same_origin),
        "source_splits_per_group": 5, "intervals_recomputed": False,
        "existing_interval_family_size_unchanged": 84,
        "symmetric_color_limits_pp": [-limit, limit], "printed_width_tex_pt": 372,
        "colormap": cmap.name, "positive_color": "blue", "negative_color": "warm red",
        "displayed_gain_pp": [[f"{x:+.2f}" for x in row] for row in grid],
        "displayed_baseline_percent": [f"{x:.2f}" for x in baseline],
        "minimum_font_size_pt": min(t.get_fontsize() for t in fig.findobj(mpl.text.Text)
                                     if t.get_text().strip()),
        "style_audit_issues": issues,
    }
    if issues:
        raise AssertionError(issues)
    style.save(fig, str(HERE / "rq3_matrix_overview"), formats=("pdf", "svg", "png"))
    (HERE / "rq3_matrix_overview_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("group_count", "same_origin_count", "model_switch_count",
                                          "minimum_font_size_pt", "style_audit_issues")}, indent=2))

if __name__ == "__main__":
    main()
