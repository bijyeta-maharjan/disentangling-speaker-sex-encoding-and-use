"""Run the probing grid and write a tidy long-format results table.

Grid: attribute x feature-set(model,block) x probe-type(linear/mlp), each with a
real-label fit and a random-label control fit (-> selectivity).

    python scripts/run_probes.py [--config CONFIG]

Outputs to <runs_dir>/probe_<sha>_<timestamp>/:
    probe_results.csv / .json   long table (one row per real-label evaluation)
    run_meta.json               provenance
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.probing import probe as P
from src.probing.extract import BLOCK_NAMES, RAW, TRAINED, UNTRAINED, load_features, load_labels
from src.probing.labels import ATTR_COLUMN
from src.probing.splits import generalizable_split, identity_split, load_si_spec
from src.utils.config import load_config, resolve
from src.utils.experiment import git_sha

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "day3_probing.yaml"

# feature sets, in plot order: (model, block, depth-index for the curve)
FEATURE_SETS = (
    [(TRAINED, b, i + 1) for i, b in enumerate(BLOCK_NAMES)]
    + [(UNTRAINED, b, i + 1) for i, b in enumerate(BLOCK_NAMES)]
    + [(RAW, "raw", 0)]
)


def feature_matrix(cfg, model, block):
    return load_features(cfg, model, None if model == RAW else block)


def run(cfg: dict) -> Path:
    labels = load_labels(cfg)
    spk_full = labels["speaker_id"].to_numpy()
    si_spec = load_si_spec(cfg)
    standardize = cfg["probe"]["standardize"]

    rows = []
    for attr in cfg["attributes"]:
        name, kind = attr["name"], attr["kind"]
        col = ATTR_COLUMN[name]

        if kind == "identity":
            train_pos, test_pos = identity_split(
                labels, cfg["identity_split"]["test_size"], cfg["seed"]
            )
            valid = labels.index.to_numpy()
        else:
            train_pos, test_pos = generalizable_split(labels, col, si_spec)
            valid = labels.index[labels[col].notna()].to_numpy()

        # encode labels over the VALID rows only, then scatter into a full array
        y_full = np.full(len(labels), -1, dtype=np.int64)
        y_valid, classes = P.encode_labels(labels.loc[valid, col].tolist())
        y_full[valid] = y_valid

        # random-label control task (consistent across feature sets)
        y_rand = P.make_random_labels(y_full, spk_full, np.concatenate([train_pos, test_pos]),
                                      kind, seed=cfg["seed"] + 100)

        print(
            f"\n=== {name} ({kind}) | classes={classes} | "
            f"train={len(train_pos)} test={len(test_pos)} "
            f"train_spk={labels.loc[train_pos,'speaker'].nunique()} "
            f"test_spk={labels.loc[test_pos,'speaker'].nunique()} ==="
        )

        for model, block, depth in FEATURE_SETS:
            X = feature_matrix(cfg, model, block)
            for probe_kind in ("linear", "mlp"):
                real = P.fit_eval(X, y_full, spk_full, train_pos, test_pos,
                                  probe_kind, cfg, standardize)
                rand = P.fit_eval(X, y_rand, spk_full, train_pos, test_pos,
                                  probe_kind, cfg, standardize)
                selectivity = real["acc"] - rand["acc"]
                rows.append({
                    "attribute": name, "kind": kind, "model": model,
                    "block": block, "depth": depth, "probe": probe_kind,
                    "acc": round(real["acc"], 5),
                    "acc_speaker_mean": round(real["acc_speaker_mean"], 5),
                    "ci_lo": round(real["ci_lo"], 5), "ci_hi": round(real["ci_hi"], 5),
                    "chance": round(real["chance"], 5),
                    "random_acc": round(rand["acc"], 5),
                    "selectivity": round(selectivity, 5),
                    "n_classes": len(classes),
                    "n_train": real["n_train"], "n_test": real["n_test"],
                    "n_test_speakers": real["n_test_speakers"],
                })
                print(
                    f"  {model:9s} {block:5s} {probe_kind:6s} | "
                    f"acc={real['acc']:.3f} chance={real['chance']:.3f} "
                    f"rand={rand['acc']:.3f} sel={selectivity:+.3f}"
                )

    df = pd.DataFrame(rows)

    # run folder
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = git_sha()
    run_dir = resolve(cfg["paths"]["runs_dir"]) / f"probe_{sha}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "probe_results.csv", index=False)
    df.to_json(run_dir / "probe_results.json", orient="records", indent=2)
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "git_sha": sha, "created": stamp, "seed": cfg["seed"],
            "checkpoint": str(resolve(cfg["checkpoint"]["path"])),
            "n_rows": len(df), "attributes": [a["name"] for a in cfg["attributes"]],
        }, fh, indent=2)
    # stable pointer to the latest run
    with open(resolve(cfg["paths"]["runs_dir"]) / "latest.txt", "w", encoding="utf-8") as fh:
        fh.write(str(run_dir))
    print(f"\n[probe] wrote {len(df)} rows -> {run_dir}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the demographic probing grid.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
