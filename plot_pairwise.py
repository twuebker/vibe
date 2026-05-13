# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "scipy",
#     "seaborn",
#     "scikit-learn",
#     "h5py",
#     "numpy",
#     "matplotlib",
# ]
# ///

"""
Grid plot: PC1/PC2 scatter + pairwise distance distribution (Brin 1995 style).

For each dataset, plots two panels:
  Left:  PC1 vs PC2 scatter (train = blue, queries = orange)
  Right: Distribution of sampled pairwise distances, normalised by the mean.
         Low coefficient of variation (cv = std/mean) indicates a concentrated
         distance distribution, which makes nearest-neighbour search harder
         (Brin 1995, "Near Neighbor Search in Large Metric Spaces").
"""

import argparse
import math
import os
import pathlib

import h5py
import matplotlib as mpl
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

# Dataset membership lists (same as plot.py)
ID_DATASETS = [
    "agnews-mxbai-1024-euclidean",
    "arxiv-nomic-768-normalized",
    "gooaq-distilroberta-768-normalized",
    "imagenet-clip-512-normalized",
    "landmark-nomic-768-normalized",
    "yahoo-minilm-384-normalized",
]
ID_DATASETS_ADDITIONAL = [
    "ccnews-nomic-768-normalized",
    "celeba-resnet-2048-cosine",
    "codesearchnet-jina-768-cosine",
    "glove-200-cosine",
    "landmark-dino-768-cosine",
    "simplewiki-openai-3072-normalized",
    "coco-i2i-512-angular",
    "deep-image-96-angular",
    "fashion-mnist-784-euclidean",
    "gist-960-euclidean",
    "glove-100-angular",
    "mnist-784-euclidean",
    "nytimes-256-angular",
    "sift-128-euclidean",
]
OOD_DATASETS = [
    "coco-nomic-768-normalized",
    "laion-clip-512-normalized",
    "llama-128-ip",
    "imagenet-align-640-normalized",
    "yandex-200-cosine",
    "yi-128-ip",
    "coco-t2i-512-angular",
]


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def find_hdf5(dataset: str, data_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the first matching HDF5 path for *dataset*, or None."""
    candidates = [
        data_dir / f"{dataset}.hdf5",
        data_dir / f"{dataset}.h5",
    ]
    if dataset == "wiki_1M":
        candidates.insert(0, data_dir / "wiki_1M_uncorrelated.h5")
    for p in candidates:
        if p.exists():
            return p
    return None


def infer_metric(dataset: str) -> str:
    """Map a dataset name to its distance metric."""
    name = dataset.lower()
    if "euclidean" in name:
        return "euclidean"
    if "angular" in name or "normalized" in name:
        return "angular"
    if "cosine" in name:
        return "cosine"
    if name.endswith("-ip") or name.endswith("-dot") or "dot" in name:
        return "ip"
    return "euclidean"


def pairwise_distances(vecs: np.ndarray, metric: str) -> np.ndarray:
    """Compute all upper-triangle pairwise distances for a sample matrix."""
    vecs = vecs.astype(np.float32)
    i_idx, j_idx = np.triu_indices(len(vecs), k=1)
    a, b = vecs[i_idx], vecs[j_idx]

    if metric == "euclidean":
        return np.sqrt(((a - b) ** 2).sum(axis=1))

    # angular / cosine: arccos of normalised dot product
    if metric in ("angular", "cosine"):
        an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)
        bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-9)
        dots = (an * bn).sum(axis=1).clip(-1.0, 1.0)
        return np.arccos(dots)

    # ip / dot: not a proper metric — fall back to euclidean
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def sample_pairwise(
    hdf5_path: pathlib.Path,
    dataset: str,
    n_vecs: int = 300,
    seed: int = 42,
) -> np.ndarray | None:
    """
    Sample *n_vecs* training vectors and return all pairwise distances.
    Indices are sorted before reading to keep HDF5 I/O sequential.
    """
    rng = np.random.default_rng(seed)
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "train" not in f:
                return None
            n_train = f["train"].shape[0]
            idx = np.sort(rng.choice(n_train, size=min(n_vecs, n_train), replace=False))
            vecs = f["train"][idx]
        return pairwise_distances(vecs, infer_metric(dataset))
    except Exception as exc:
        print(f"  Warning ({dataset}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def dataset_pairwise_grid(
    out_dir: pathlib.Path,
    pca_mahalanobis: pl.DataFrame,
    data_dir: pathlib.Path,
    datasets: list[str] | None = None,
    n_cols: int = 3,
    max_scatter: int = 2000,
    n_vecs: int = 300,
):
    """
    One row per dataset, two panels each:
      [PC1/PC2 scatter] [pairwise distance distribution]
    """
    known_order = ID_DATASETS + ID_DATASETS_ADDITIONAL + OOD_DATASETS
    available = set(pca_mahalanobis["dataset"].unique().to_list())

    if datasets is None:
        datasets = [d for d in known_order if d in available] + sorted(available - set(known_order))
    datasets = [d for d in datasets if d in available]

    n = len(datasets)
    n_rows = math.ceil(n / n_cols)
    colors = {"train": "#1f77b4", "test": "#ff7f0e"}

    fig = plt.figure(figsize=(n_cols * 4.0, n_rows * 2.5))
    outer = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    for i, dataset in enumerate(datasets):
        row, col = divmod(i, n_cols)
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.12)
        ax_pca = fig.add_subplot(inner[0])
        ax_dist = fig.add_subplot(inner[1])

        pdata = pca_mahalanobis.filter(pl.col("dataset") == dataset)

        # ---- title ----
        parts = dataset.split("-")
        if dataset.endswith("-binary"):
            title = "-".join(parts[:-3]) + "-binary"
        elif len(parts) >= 3:
            title = "-".join(parts[:-2])
        else:
            title = dataset

        # ---- PC1/PC2 scatter ----
        for part in ["train", "test"]:
            p = pdata.filter(pl.col("part") == part)
            if p.height > max_scatter:
                p = p.sample(max_scatter, seed=42)
            ax_pca.scatter(
                p["x"].to_numpy(),
                p["y"].to_numpy(),
                s=0.3,
                alpha=0.6,
                color=colors[part],
                rasterized=True,
            )
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        ax_pca.set_xlabel("PC1", fontsize=6, labelpad=2)
        ax_pca.set_ylabel("PC2", fontsize=6, labelpad=2)
        ax_pca.set_title(title, fontsize=7, pad=2)

        # ---- pairwise distance distribution ----
        hdf5_path = find_hdf5(dataset, data_dir)
        if hdf5_path is not None:
            print(f"  {dataset}: loading from {hdf5_path.name}")
            dists = sample_pairwise(hdf5_path, dataset, n_vecs=n_vecs)
        else:
            dists = None
            print(f"  {dataset}: no HDF5 found, skipping distance plot")

        if dists is not None and len(dists) > 10:
            mean_d = float(dists.mean())
            cv = float(dists.std() / mean_d) if mean_d > 0 else float("nan")
            norm = dists / mean_d if mean_d > 0 else dists

            ax_dist.hist(norm, bins=60, density=True, color="#1f77b4", alpha=0.75, linewidth=0)
            ax_dist.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.8)
            ax_dist.set_xlabel(f"d / E[d]   cv={cv:.3f}", fontsize=6, labelpad=2)
        else:
            ax_dist.text(
                0.5, 0.5,
                "no HDF5" if hdf5_path is None else "no data",
                ha="center", va="center",
                transform=ax_dist.transAxes,
                fontsize=7, color="gray",
            )

        for ax in (ax_pca, ax_dist):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.5)

    # blank out unused cells
    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.12)
        for j in range(2):
            fig.add_subplot(inner[j]).axis("off")

    legend_handles = [
        Patch(color=colors["train"], label="data"),
        Patch(color=colors["test"], label="queries"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0), fontsize=8)

    out_path = out_dir / "dataset-pairwise-grid.pdf"
    print("Writing", out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot PC1/PC2 scatter + pairwise distance distribution for all datasets."
    )
    ap.add_argument("--results", default="results", help="directory containing data-pca-mahalanobis.parquet")
    ap.add_argument("--data", default="data", help="directory containing HDF5 dataset files")
    ap.add_argument("--output", default="plots", help="output directory")
    ap.add_argument("--n-vecs", type=int, default=300,
                    help="number of training vectors to sample per dataset (yields ~n*(n-1)/2 pairs)")
    ap.add_argument("--n-cols", type=int, default=3, help="grid columns")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated list of datasets (default: all present in parquet)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results_dir = pathlib.Path(args.results)
    data_dir = pathlib.Path(args.data)
    out_dir = pathlib.Path(args.output)

    pca_mahalanobis = pl.read_parquet(results_dir / "data-pca-mahalanobis.parquet")
    datasets = args.datasets.split(",") if args.datasets else None

    dataset_pairwise_grid(
        out_dir,
        pca_mahalanobis,
        data_dir,
        datasets=datasets,
        n_cols=args.n_cols,
        n_vecs=args.n_vecs,
    )
