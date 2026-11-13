"""Render frozen source-test sensitivity results at native manuscript width.

uv run --isolated --no-project --with matplotlib==3.11.2 --with numpy==2.5.4 \
    python build_figure.py
Only categorical vertical placement changes to separate markers. All numerical
x coordinates are read from the parent supplement; no interval estimation.
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
HEADER = "#245C78"
POINT_SIZE = 14
POINT_STROKE = .7
LANE_PITCH = 5.2  # physical points, including stroke and visible clearance


def pack(records, xlim, plot_width):
    """Allocate nonoverlapping lanes without changing numerical x values."""
    lanes, offsets = [], {}
    for row in sorted(records, key=lambda r: (r["value_pp"], r["cell_id"])):
        xp = (row["value_pp"] - xlim[0]) * plot_width / (xlim[1] - xlim[0])
        lane = next((i for i, xs in enumerate(lanes)
                     if all(abs(xp - x) >= LANE_PITCH for x in xs)), len(lanes))
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append(xp)
        offsets[row["cell_id"]] = 10 + lane * LANE_PITCH
    return offsets, 10 + len(lanes) * LANE_PITCH + 6


def marker_bounds(ax, collection, x, y, kind):
    """Bounds in physical points, including the marker edge stroke."""
    center = ax.transData.transform((x, y)) * 72 / ax.figure.dpi
    path = collection.get_paths()[0].get_extents()
    size = np.sqrt(collection.get_sizes()[0])
    stroke = collection.get_linewidths()[0]
    return {"kind": kind, "x": float(center[0]), "y": float(center[1]),
            "width": float(path.width * size + stroke),
            "height": float(path.height * size + stroke)}


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
    width_pt, left, right = WIDTH * 72, 72, 40
    plot_width = width_pt - left - right
    panels = [
        {"id": "a", "views": VIEWS[1:], "xlim": (-4.3, 2),
         "ticks": [-4, -3, -2, -1, 0, 1, 2], "reference": 0,
         "title": "(a) Macro-F1 change", "xlabel": "Change from original (pp)",
         "stat": "mean", "column": "Mean"},
        {"id": "b", "views": VIEWS, "xlim": (-.2, 4.6),
         "ticks": [0, 1, 2, 3, 4], "reference": 1,
         "title": "(b) Selection gap", "xlabel": "Macro-F1 gap (pp)",
         "stat": "median", "column": "Median"},
    ]
    plotted, placements = [], []
    for panel in panels:
        cursor, groups = 4.0, []
        for view in panel["views"]:
            records = []
            for row in originals:
                current = index[row["cell_id"], view]
                value = (float(current["pstar_macro_f1_pp"]) - float(row["pstar_macro_f1_pp"])
                         if panel["id"] == "a" else float(current["selection_gap_pp"]))
                records.append({"panel": panel["id"], "cell_id": row["cell_id"],
                                "model": row["model"], "optimizer": row["optimizer"],
                                "seed": row["seed"], "view": view, "value_pp": value})
            values = [r["value_pp"] for r in records]
            value = float(np.mean(values) if panel["id"] == "a" else np.median(values))
            expected = summary["source_test_sensitivity"][view][
                "pstar_macro_f1_pp_change_from_original" if panel["id"] == "a" else "selection_gap_pp"
            ][panel["stat"]]
            assert abs(value - expected) < 1e-12
            assert min(values) > panel["xlim"][0] and max(values) < panel["xlim"][1]
            offsets, group_height = pack(records, panel["xlim"], plot_width)
            groups.append({"view": view, "summary": value, "y": cursor,
                           "height": group_height, "records": records, "offsets": offsets})
            plotted.extend(records)
            cursor += group_height + 4
        panel.update(groups=groups, body=cursor)
    top, between, bottom = 60, 62, 33
    height = top + panels[0]["body"] + between + panels[1]["body"] + bottom
    fig = plt.figure(figsize=(WIDTH, height / 72))
    panel_top = height - top
    geometry = []
    for panel in panels:
        body = panel["body"]
        y_bottom = panel_top - body
        ax = fig.add_axes([left / width_pt, y_bottom / height,
                           plot_width / width_pt, body / height])
        ax.set_ylim(body, 0)
        ax.set_xlim(panel["xlim"])
        ax.set_xticks(panel["ticks"])
        ys, labels, markers = [], [], []
        for group in panel["groups"]:
            y, gh = group["y"], group["height"]
            ax.axhspan(y - 3, y + gh - 2, color="#F5F8FA", zorder=0)
            ys.append(y + (gh - 2) / 2)
            labels.append(LABELS[VIEWS.index(group["view"])])
            value = group["summary"]
            mark = ax.scatter(value, y, marker="D", s=22, color="#222222",
                              edgecolor="white", linewidth=.6, zorder=5)
            markers.append((mark, value, y, "summary"))
            fig.text(.988, (panel_top - y) / height,
                     f"{value:+.3f}" if panel["id"] == "a" else f"{value:.3f}",
                     ha="right", va="center", fontsize=8)
            for row in group["records"]:
                mi = MODELS.index(row["model"])
                cy = y + group["offsets"][row["cell_id"]]
                mark = ax.scatter(row["value_pp"], cy, s=POINT_SIZE, marker=MARKERS[mi],
                                  facecolor=COLORS[mi] if row["optimizer"] == "pe2" else "white",
                                  edgecolor=COLORS[mi], linewidth=POINT_STROKE, zorder=3)
                markers.append((mark, row["value_pp"], cy, "cell"))
                placements.append({**row, "y_physical_pt": cy,
                                   "summary_y_physical_pt": y})
        ax.set_yticks(ys, labels)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.set_xlabel(panel["xlabel"], labelpad=5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True, color="#DCE2E7", lw=.4)
        ax.spines["left"].set_visible(False)
        ax.axvline(panel["reference"], color="#727A82", ls="--", lw=.75, zorder=1)
        fig.text(.01, (panel_top + 8) / height, panel["title"],
                 fontsize=8.5, fontweight="bold", color=HEADER, va="bottom")
        fig.text(.988, (panel_top + 8) / height, panel["column"],
                 ha="right", color=HEADER, va="bottom")
        geometry.append((panel["id"], ax, markers))
        panel_top = y_bottom - between

    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls="", label=n, markersize=4)
                        for c, m, n in zip(COLORS, MARKERS, NAMES)],
               loc="upper center", bbox_to_anchor=(.53, 1), ncol=3,
               handletextpad=.5, columnspacing=1.3)
    fig.legend(handles=[Line2D([], [], color="#555555", marker="o", ls="", label="Filled: PE2", markersize=4),
                        Line2D([], [], mec="#555555", mfc="white", marker="o", ls="", label="Open: BT-APE", markersize=4),
                        Line2D([], [], color="#222222", marker="D", ls="", label="Mean (a); median (b)", markersize=4)],
               loc="upper center", bbox_to_anchor=(.53, 1 - 17 / height), ncol=3,
               handletextpad=.5, columnspacing=1.1)
    fig.canvas.draw()
    clearance = {}
    for panel, ax, markers in geometry:
        bounds = [marker_bounds(ax, *item) for item in markers]
        gaps = [max(abs(a["x"] - b["x"]) - (a["width"] + b["width"]) / 2,
                    abs(a["y"] - b["y"]) - (a["height"] + b["height"]) / 2)
                for i, a in enumerate(bounds) for b in bounds[i + 1:]]
        assert min(gaps) >= .5, (panel, min(gaps))
        clearance[panel] = {"minimum_marker_box_clearance_pt": min(gaps),
                            "cell_markers": sum(b["kind"] == "cell" for b in bounds),
                            "summary_markers": sum(b["kind"] == "summary" for b in bounds)}
    issues = style._audit(fig)
    report = {"issues": issues, "printed_width_inches": WIDTH,
              "printed_height_inches": height / 72,
              "matplotlib_version": mpl.__version__, "numpy_version": np.__version__,
              "font": style._resolved_font, "plotted_cell_values": len(plotted),
              "summary_matches_audit": True, "marker_geometry_checks": clearance,
              "packing": "Only categorical y positions change; x values are exact.",
              "input_sha256": {n: hashlib.sha256((DATA / n).read_bytes()).hexdigest()
                               for n in ["cell-test-sensitivity.csv", "audit-results.json"]}}
    (HERE / "figure-checks.json").write_text(json.dumps(report, indent=2) + "\n")
    if issues:
        fig.savefig(HERE / "layout-review.png", dpi=180)
        raise AssertionError(issues)
    style.save(fig, str(HERE / "Fig11-source-duplicates"), formats=("pdf", "svg", "png"))
    for name, records in [("plotted-values.csv", plotted), ("point-placements.csv", placements)]:
        with (HERE / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
