"""Compute acoustic descriptors over a subsample of utterances per speaker.

Goal: SEE distributions, not exhaust the dataset.
Default: ~10 utterances per speaker (~600 files total).

Cached to ``data/processed/acoustic_features.parquet``; re-running is a no-op
unless ``--force`` is passed. Use ``--per-speaker N`` to grow the sample
(or 0 for everything).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.acoustic import compute_features  # noqa: E402
from src.data.index import build_manifest  # noqa: E402
from src.data.metadata import load_metadata  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

DEFAULT_PER_SPEAKER = 10
DEFAULT_CACHE = resolve("data/processed/acoustic_features.parquet")


def subsample(manifest: pd.DataFrame, per_speaker: int, seed: int) -> pd.DataFrame:
    if per_speaker <= 0:
        return manifest.copy()
    return (
        manifest.groupby("speaker", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_speaker), random_state=seed))
        .reset_index(drop=True)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-speaker", type=int, default=DEFAULT_PER_SPEAKER)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.cache.exists() and not args.force:
        print(f"Cache exists at {args.cache} — skipping. Pass --force to recompute.")
        return 0

    args.cache.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    meta = load_metadata()
    sample = subsample(manifest, args.per_speaker, args.seed)
    print(
        f"Computing features for {len(sample)} utterances "
        f"({args.per_speaker}/speaker over {sample['speaker'].nunique()} speakers)..."
    )

    records = []
    for row in tqdm(sample.itertuples(index=False), total=len(sample)):
        try:
            feats = compute_features(row.filepath)
        except Exception as e:
            print(f"[skip] {row.filepath}: {e}")
            continue
        records.append(
            {
                "filepath": row.filepath,
                "speaker": row.speaker,
                "digit": row.digit,
                "repetition": row.repetition,
                **feats,
            }
        )

    df = pd.DataFrame(records)
    df = df.merge(meta, on="speaker", how="left")
    df.to_parquet(args.cache, index=False)
    print(f"Wrote {len(df)} rows -> {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
