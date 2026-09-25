#!/usr/bin/env python3
"""Render the completed repeat grid as mean +/- SEM, without model calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_extraction(out):
    """Render the extraction results on development and official test."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = pd.read_csv(out / "extraction_seeds.csv")
    for split in ("dev", "final"):
        for body in ("python", "regex"):
            for sample in (20, 55, 110):
                group = rows[(rows.condition == f"extract_{body}_M{sample}") & (rows.split == split)]
                assert sorted(group.seed) == manifest["seeds"], (split, body, sample)
        assert len(rows[(rows.condition == "extract_pool") & (rows.split == split)]) == 1
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.2,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for axis, split, title in zip(axes, ("dev", "final"), ("Development set", "Official test")):
        for body, color, marker in [("python", "#0072B2", "o"), ("regex", "#D55E00", "s")]:
            group = rows[rows.condition.str.startswith("extract_" + body) & (rows.split == split)]
            aggregate = group.groupby("M").micro_f1.agg(["mean", "sem"]).sort_index()
            axis.errorbar(aggregate.index, aggregate["mean"], yerr=aggregate["sem"],
                          label="CLIPPER " + body.title(), color=color, marker=marker,
                          markersize=3, linewidth=1.2, capsize=2)
        pool = rows[(rows.condition == "extract_pool") & (rows.split == split)].iloc[0]
        axis.axhline(pool.micro_f1, color="#202020", linestyle=":", label="Hand-written pool")
        axis.set_title(title, loc="left", fontsize=9)
        axis.set_xticks([20, 55, 110])
        axis.set_xlabel("Prompt sample cap M")
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Extraction micro F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(.5, .075))
    fig.text(.5, .01, "Ten most frequent training keys; mean +/- SEM over ten seeds; fixed evaluation rows",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .2, 1, 1))
    destination = out / "figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"extraction-repeats.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote completed extraction figure under {destination}")


def plot(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    cls = pd.read_csv(out / "classification_seeds.csv")
    ext = pd.read_csv(out / "extraction_seeds.csv")
    class_names = {"slipper_full": "SLIPPER full", "slipper_M20": "SLIPPER M20",
                   "slopper_raw_M20": "CLIPPER raw M20", "slopper_raw_M100": "CLIPPER raw M100"}
    for name in class_names:
        for cap in manifest["caps"]:
            for split in ("dev", "final"):
                rows = cls[(cls.condition == name) & (cls.cap == cap) & (cls.split == split)]
                assert sorted(rows.seed) == manifest["seeds"], (name, cap, split)
    for body in ("python", "regex"):
        for m in (20, 55, 110):
            for split in ("dev", "final"):
                rows = ext[(ext.condition == f"extract_{body}_M{m}") & (ext.split == split)]
                assert sorted(rows.seed) == manifest["seeds"]
    for split in ("dev", "final"):
        assert len(ext[(ext.condition == "extract_pool") & (ext.split == split)]) == 1
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8.2,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    for split in ("dev", "final"):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
        for (name, label), color, marker in zip(class_names.items(),
                ["#202020", "#888888", "#0072B2", "#009E73"], ["o", "s", "^", "D"]):
            rows = cls[(cls.condition == name) & (cls.split == split)]
            agg = rows.groupby("cap").accuracy.agg(["mean", "sem"]).sort_index()
            axes[0].errorbar(agg.index, agg["mean"], yerr=agg["sem"], label=label,
                            color=color, marker=marker, markersize=3, linewidth=1.2, capsize=2,
                            linestyle="--" if name == "slipper_M20" else "-")
        axes[0].set_xscale("log", base=4)
        axes[0].set_xticks(manifest["caps"], [str(c) for c in manifest["caps"]])
        axes[0].set_xlabel("Maximum training rows / calculator")
        axes[0].set_ylabel(f"Exact {'dev' if split == 'dev' else 'official-test'} accuracy (%)")
        axes[0].set_title("(a) Calculator prediction", loc="left", fontsize=9)
        axes[0].legend(frameon=False, fontsize=7)
        for body, color, marker in [("python", "#0072B2", "o"), ("regex", "#D55E00", "s")]:
            rows = ext[(ext.condition.str.startswith("extract_" + body)) & (ext.split == split)]
            agg = rows.groupby("M").micro_f1.agg(["mean", "sem"]).sort_index()
            axes[1].errorbar(agg.index, agg["mean"], yerr=agg["sem"], label="CLIPPER " + body.title(),
                            color=color, marker=marker, markersize=3, linewidth=1.2, capsize=2)
        pool = ext[(ext.condition == "extract_pool") & (ext.split == split)].iloc[0]
        axes[1].axhline(pool.micro_f1, color="#202020", linestyle=":", label="Hand-written pool")
        axes[1].set_xticks([20, 55, 110])
        axes[1].set_xlabel("Prompt sample cap M")
        axes[1].set_ylabel(f"{'Dev' if split == 'dev' else 'Official-test'} micro F1 (%)")
        axes[1].set_title("(b) Value extraction: ten keys", loc="left", fontsize=9)
        axes[1].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.grid(axis="y", alpha=.2)
        fig.text(.5, .01, "Mean +/- SEM over ten training seeds; fixed evaluation rows", ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .06, 1, 1))
        for suffix in ("pdf", "png"):
            fig.savefig(figures / f"medcalc-{split}-repeats.{suffix}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    # Per-seed contrasts reuse the same evaluation rows; no keys are treated as seeds.
    contrasts = []
    for split in ("dev", "final"):
        pool = ext[(ext.condition == "extract_pool") & (ext.split == split)].iloc[0]
        for name, rows in ext[(ext.condition != "extract_pool") & (ext.split == split)].groupby("condition"):
            for metric in ("micro_f1", "exact_set"):
                delta = rows[metric] - pool[metric]
                contrasts.append(dict(split=split, condition=name, reference="extract_pool", metric=metric,
                    n_seeds=len(delta), mean_difference=delta.mean(), sem_difference=delta.sem()))
    pd.DataFrame(contrasts).to_csv(out / "extraction_pool_contrasts.csv", index=False)
    lines = ["# MedCalc repeated experiments", "", "All values are percentages; uncertainty is SEM over ten training seeds.",
             "The pool is a deterministic reference. Evaluation sets are fixed.", "",
             "Extraction keys: " + ", ".join(manifest["extraction_keys"]) + ".", ""]
    for split, title in [("dev", "Development calculator accuracy"),
                         ("final", "Official-test calculator accuracy")]:
        lines += ["## " + title, "", "| Cap | " + " | ".join(class_names.values()) + " |",
                  "|---|---:|---:|---:|---:|"]
        for cap in manifest["caps"]:
            values = []
            for condition in class_names:
                rows = cls[(cls.condition == condition) & (cls.cap == cap) & (cls.split == split)]
                values.append(f"{rows.accuracy.mean():.2f} ± {rows.accuracy.sem():.2f}")
            lines.append("| " + str(cap) + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Official-test extraction", "", "| Method | Precision | Recall | F1 | All selected fields |",
              "|---|---:|---:|---:|---:|"]
    for name, rows in ext[ext.split == "final"].groupby("condition", sort=True):
        values = []
        for metric in ("micro_p", "micro_r", "micro_f1", "exact_set"):
            values.append(f"{rows[metric].mean():.2f}" if name == "extract_pool"
                          else f"{rows[metric].mean():.2f} ± {rows[metric].sem():.2f}")
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines += ["", "## Interpretation", "",
              "These repeats vary local learner seeds on fixed training rows. The provider seed and temperature are unset.",
              "Runtime exceptions retain the existing abstention semantics and are recorded per split in each cell result.",
              "The ten-key results have a different denominator from the historical 34-key extraction results.",
              "SEM bars and non-significant differences alone do not establish practical equivalence.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote complete ten-seed figures and tables under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extraction-only", action="store_true")
    args = parser.parse_args()
    (plot_extraction if args.extraction_only else plot)(args.out)
