"""Build the layer-wise (cnn1d) probing figures + the terminal summary from cached
features and the latest probe results.

    python figures/plot_layerwise_probes.py [--config CONFIG] [--run RUN_DIR]

    # paper figures only, from the saved results (no dataset or caches needed):
    python figures/plot_layerwise_probes.py --decodability-only --run results/probing_layerwise_cnn1d

Produces:
    figures/output/day3_decodability_<attr>.png   (3 lines + chance)
    figures/output/day3_embed_<block>.png          (PCA/t-SNE x digit/sex/speaker)
    <run_dir>/cluster_mi.json                     (KMeans-vs-attribute MI)
    <run_dir>/day3_summary.json                   (peak / margin / trend per attr)
and prints a concise summary.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probing.extract import load_features, load_labels
from src.utils.config import load_config, resolve
from src.viz.decodability import BLOCK_ORDER, plot_decodability
from src.viz.embeddings import cluster_mi, compute_embeddings, plot_embedding_grid

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "day3_probing.yaml"


def latest_run(cfg: dict) -> Path:
    ptr = resolve(cfg["paths"]["runs_dir"]) / "latest.txt"
    return Path(ptr.read_text(encoding="utf-8").strip())


def build_decodability(df, cfg, figs_dir) -> None:
    for attr in df["attribute"].unique():
        out = figs_dir / f"day3_decodability_{attr}.png"
        plot_decodability(df, attr, probe="linear", save_path=out)
        print(f"[fig] {out}")


def build_embeddings(cfg, run_dir, figs_dir) -> dict:
    labels = load_labels(cfg)
    mi_report = {}
    for block in cfg["viz"]["embedding_blocks"]:
        X = load_features(cfg, "trained", block)
        emb = compute_embeddings(X, cfg, seed=cfg["seed"])
        labels_sub = labels.iloc[emb["sel"]].reset_index(drop=True)
        out = figs_dir / f"day3_embed_{block}.png"
        plot_embedding_grid(emb, labels_sub, block, cfg["viz"]["color_by"], save_path=out)
        print(f"[fig] {out}  (PCA 2D var={emb['pca_evr_2d']:.1%}, "
              f"{len(emb['pca_evr'])}-comp cum var={emb['pca_evr_cum']:.1%})")
        if cfg["viz"]["cluster_mi"]["enabled"] and block == "conv5":
            mi_report[block] = cluster_mi(
                X, labels_sub, emb["sel"],
                cfg["viz"]["cluster_mi"]["n_clusters"], cfg["seed"],
            )
    if mi_report:
        with open(run_dir / "cluster_mi.json", "w", encoding="utf-8") as fh:
            json.dump(mi_report, fh, indent=2)
        print(f"[mi] {run_dir / 'cluster_mi.json'}")
    return mi_report


def summarize(df, mi_report, run_dir) -> dict:
    """Per attribute: peak trained acc (block), trained-untrained margin, depth trend."""
    summary = {}
    lin = df[df["probe"] == "linear"]
    for attr in lin["attribute"].unique():
        a = lin[lin["attribute"] == attr]
        tr = a[a["model"] == "trained"].set_index("block").reindex(BLOCK_ORDER)
        un = a[a["model"] == "untrained"].set_index("block").reindex(BLOCK_ORDER)
        raw = a[a["model"] == "raw"]
        peak_block = tr["acc"].idxmax()
        peak = float(tr["acc"].max())
        un_at_peak = float(un.loc[peak_block, "acc"])
        chance = float(a["chance"].iloc[0])
        # depth trend: conv5 - conv1 on trained
        trend = float(tr.loc["conv5", "acc"] - tr.loc["conv1", "acc"])
        summary[attr] = {
            "peak_trained_acc": round(peak, 4),
            "peak_block": peak_block,
            "untrained_at_peak": round(un_at_peak, 4),
            "trained_minus_untrained": round(peak - un_at_peak, 4),
            "selectivity_at_peak": round(float(tr.loc[peak_block, "selectivity"]), 4),
            "chance": round(chance, 4),
            "raw_acc": round(float(raw["acc"].iloc[0]), 4) if not raw.empty else None,
            "depth_trend_conv5_minus_conv1": round(trend, 4),
            "trained_by_block": {b: round(float(tr.loc[b, "acc"]), 4) for b in BLOCK_ORDER},
        }
    out = {"per_attribute": summary, "cluster_mi": mi_report}
    with open(run_dir / "day3_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return summary


def print_summary(summary, mi_report) -> None:
    print("\n" + "=" * 72)
    print("LAYER-WISE DEMOGRAPHIC DECODABILITY SUMMARY (linear probe, trained SI)")
    print("=" * 72)
    for attr, s in summary.items():
        trend = s["depth_trend_conv5_minus_conv1"]
        arrow = "rises with depth" if trend > 0.03 else ("falls with depth" if trend < -0.03 else "flat with depth")
        print(f"\n{attr}:")
        print(f"  peak decodability (trained) : {s['peak_trained_acc']:.3f} @ {s['peak_block']} "
              f"(chance {s['chance']:.3f}, raw {s['raw_acc']})")
        print(f"  trained - untrained @ peak  : {s['trained_minus_untrained']:+.3f} "
              f"(untrained {s['untrained_at_peak']:.3f})")
        print(f"  selectivity @ peak          : {s['selectivity_at_peak']:+.3f}")
        print(f"  depth trend (conv5 - conv1) : {trend:+.3f}  -> {arrow}")
    if mi_report:
        print("\ncluster MI (conv5, KMeans):")
        for blk, m in mi_report.items():
            for attr in ("digit", "sex", "speaker_id"):
                print(f"  {blk} vs {attr:11s}: NMI={m[attr]['nmi']:.3f} AMI={m[attr]['ami']:.3f}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build layer-wise probing figures + summary.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--run", default=None, help="probe run dir (default: latest)")
    ap.add_argument("--decodability-only", action="store_true",
                    help="only redraw the decodability curves from probe_results.csv "
                         "(skips embeddings, which need the activation caches)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = resolve(args.run) if args.run else latest_run(cfg)
    df = pd.read_csv(run_dir / "probe_results.csv")
    figs_dir = resolve(cfg["paths"]["figs_dir"])
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[layerwise] using probe run: {run_dir}")
    build_decodability(df, cfg, figs_dir)
    if args.decodability_only:
        return
    mi_report = build_embeddings(cfg, run_dir, figs_dir)
    summary = summarize(df, mi_report, run_dir)
    print_summary(summary, mi_report)


if __name__ == "__main__":
    main()
