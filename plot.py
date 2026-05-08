# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "scipy",
#     "seaborn",
#     "scikit-learn",
#     "networkx"
# ]
# ///

import pathlib
import os
import sys
import math
import argparse
import itertools
from scipy.stats import wilcoxon
import polars as pl
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from vibe.definitions import get_definitions
from vibe.main import filter_disabled_algorithms, filter_algorithms_by_device

# The list of in-distribution datasets
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
]
# The list of out of distribution datasets
OOD_DATASETS = [
    "coco-nomic-768-normalized",
    "laion-clip-512-normalized",
    "llama-128-ip",
    "imagenet-align-640-normalized",
    "yandex-200-cosine",
    "yi-128-ip",
]
NEW_DATASETS = {"arxiv_1M", "wiki_1M", "yfcc_1M"}

sns.set_palette("tab10")


def radar_chart(
    data,
    theta,
    radius,
    ticks,
    ax=None,
    smooth=False,
    show_percentiles=False,
    shorten_labels=False,
    supporting_ip=True,
    theta_offset=0,
    **kwargs,
):
    from scipy.interpolate import make_interp_spline
    import numpy as np

    # Enforce the order of the datasets
    data = data.set_index("dataset").loc[ticks].reset_index()

    categories = data[theta].to_list()
    values = data[radius].to_list()
    types = data["dataset-type"].to_list()

    num_vars = len(categories)
    theta = np.linspace(0, 2 * np.pi, num_vars + 1, endpoint=True)
    values.append(values[0])

    if ax is None:
        ax = plt.gca()

    ax.set_theta_offset(theta_offset)

    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    datatype_palette = {"in-distribution": "tab:blue", "out-of-distribution": "tab:orange"}

    if smooth:
        theta_smooth = np.linspace(0, 2 * np.pi, 1000)
        values_smooth = make_interp_spline(theta, values, bc_type="periodic", k=3)(theta_smooth)
        ax.plot(theta_smooth, values_smooth, color="gray")
        ax.fill_between(theta_smooth, values_smooth, color="gray", alpha=0.3)
    else:
        ax.plot(theta, values, color="gray")
        ax.fill_between(theta, values, color="gray", alpha=0.3)

    for t, val, data_type in zip(theta, values, types):
        if val > 0.0:
            color = datatype_palette.get(data_type, "red")
            ax.scatter([t], [val], c=color, zorder=10)

    for ytick in yticks:
        ax.add_patch(plt.Circle((0, 0), ytick, transform=ax.transData._b, color="gray", alpha=0.1))

    for data_type, t, dataset in zip(types, theta[:-1], categories):
        if supporting_ip or "-ip" not in dataset:
            color = datatype_palette.get(data_type, "red")
            ax.axvline(t, c=color, linewidth=1, alpha=0.6, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    if show_percentiles:
        for t in yticks[:-1]:
            ax.annotate(xy=(np.pi / 2 - theta_offset, t), text=f"{t * 100}%", fontsize=7, va="center")
    if shorten_labels:
        xlabels = [t[:2] for t in ticks]
    else:
        xlabels = [t.split("-")[0] for t in ticks]
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(xlabels)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(theta[:-1])
    ax.grid(False)


def fastest_at(data, recall=0.9, k=100):
    return (
        data.filter(pl.col("k") == k)
        .filter(pl.col("recall") >= recall)
        .with_columns(pl.col("qps").rank(descending=True).over("dataset", "algorithm").alias("__tmp__"))
        .filter(pl.col("__tmp__") == 1)
        .select(pl.exclude("__tmp__"))
    )


def radar_at_recall_plot(
    out_dir,
    data,
    query_stats,
    recall,
    algorithms,
    all_algorithms,
    ncols=5,
    height=4.5,
    k=100,
    gpu=False,
):
    data = data.filter(pl.col("dataset").is_in(ID_DATASETS + OOD_DATASETS)).filter(
        pl.col("algorithm").is_in(all_algorithms)
    )
    datasets = data["dataset"].unique().to_list()
    expected_combinations = pl.DataFrame({"dataset": datasets}).join(
        pl.DataFrame({"algorithm": algorithms}), how="cross"
    )

    supports_ip = data.filter(pl.col("dataset").str.contains("-ip"))["algorithm"].unique().to_list()

    plot_data = (
        data.filter(pl.col("k") == k)
        .filter(pl.col("recall") >= recall)
        .with_columns(pl.col("qps").rank(descending=True).over("dataset", "algorithm").alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("dataset", "algorithm", "params", "recall", "qps")
        .with_columns((pl.col("qps") / pl.col("qps").max().over("dataset")).alias("qps_frac"))
        .sort("qps", descending=True)
        .join(expected_combinations, on=["algorithm", "dataset"], how="right")
        .with_columns(
            pl.when(pl.col("qps").is_not_null())
            .then(pl.col("qps").map_elements(lambda x: f"{x:.0f}", return_dtype=pl.String))
            .otherwise(pl.lit("x"))
            .alias("label"),
            pl.when(pl.col("dataset").is_in(ID_DATASETS))
            .then(pl.lit("in-distribution"))
            .when(pl.col("dataset").is_in(OOD_DATASETS))
            .then(pl.lit("out-of-distribution"))
            .otherwise(pl.lit("unknown-type"))
            .alias("dataset-type"),
        )
        .with_columns(pl.col("qps", "recall", "qps_frac").fill_null(0))
        .filter(pl.col("algorithm").is_in(algorithms))
        .select("algorithm", "dataset", "qps_frac", "dataset-type")
    )

    avg_rc = query_stats.group_by("dataset").agg(pl.col("rc100").mean())

    dataset_order = (
        plot_data.select("dataset", "dataset-type")
        .unique()
        .join(avg_rc, on=["dataset"])
        .with_columns(~pl.col("dataset").str.contains("-ip").alias("is-ip"))
        .sort("dataset-type", "is-ip", "rc100", "dataset")
    )["dataset"].to_list()

    algorithm_order = (
        plot_data.with_columns(pl.col("qps_frac").rank(descending=True).over("dataset").alias("rank"))
        .group_by("algorithm")
        .agg(pl.col("rank").mean())
        .sort("rank")
    )["algorithm"].to_list()

    width = ncols * 2.25
    fig, axs = plt.subplots(
        figsize=(width, height),
        ncols=ncols,
        nrows=math.ceil(len(algorithm_order) / ncols),
        subplot_kw=dict(projection="polar"),
    )
    axs = [ax for sub in axs for ax in sub]
    for ax in axs:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    theta_offset = -1 / 3 * math.pi
    for algo, ax in zip(algorithm_order, axs[1:]):
        facet_data = plot_data.filter(pl.col("algorithm") == algo)
        radar_chart(
            facet_data.to_pandas(),
            theta="dataset",
            radius="qps_frac",
            ticks=dataset_order,
            ax=ax,
            shorten_labels=True,
            supporting_ip=algo in supports_ip,
            theta_offset=theta_offset,
        )
        ax.set_title(algo)

    # setup the legend
    legend_data = (
        plot_data.select("dataset", "dataset-type")
        .unique()
        .sort("dataset-type", "dataset")
        .with_columns(qps_frac=pl.lit(0.0))
    )
    radar_chart(
        legend_data.to_pandas(),
        theta="dataset",
        radius="qps_frac",
        ticks=dataset_order,
        show_percentiles=True,
        ax=axs[0],
        theta_offset=theta_offset,
    )

    plt.tight_layout()
    gpu_suffix = "-gpu" if gpu else ""
    filename = f"radar-{recall}{gpu_suffix}.png"
    print("Writing", out_dir / filename)
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def compute_pareto(data, by=("algorithm", "dataset", "k"), id_col="params"):
    df = data

    if id_col not in df.columns:
        df = df.with_row_count("row_id")
        id_col = "row_id"

    dominated = (
        df.join(df, on=list(by), how="inner", suffix="_r")
        .filter(
            (pl.col("qps") <= pl.col("qps_r"))
            & (pl.col("recall") <= pl.col("recall_r"))
            & ((pl.col("qps") < pl.col("qps_r")) | (pl.col("recall") < pl.col("recall_r")))
        )
        .select([*by, id_col])
        .unique()
    )

    pareto = df.join(dominated, on=[*by, id_col], how="anti")
    return pareto


def adjust_text(texts, height):
    inv_data_transform = plt.gca().transData.inverted()
    data_transform = plt.gca().transData

    def get_x_display(text):
        return data_transform.transform(text.get_position())[0]

    def get_y_display(text):
        return data_transform.transform(text.get_position())[1]

    texts = sorted(texts, reverse=True, key=lambda text: text.get_position()[1])
    for prev, text in zip(texts, texts[1:]):
        if get_y_display(prev) - get_y_display(text) < height:
            newpos = inv_data_transform.transform((get_x_display(text), get_y_display(prev) - height))
            text.set_y(newpos[1])

        pass


def pareto_plot(
    out_dir,
    data,
    pca_mahalanobis,
    datasets,
    algorithms,
    k: int = 100,
    xlim=(0.5, 1.0),
    ylim=(2e2, 1.4e4),
    *,
    figsize=(10, 6),
    separate_legend: bool = True,
    gpu=False,
):
    def flatten_if_needed(data):
        if data and isinstance(data[0], list):
            return [item for sublist in data for item in sublist]
        else:
            return data

    threshold_recall = 0.90
    gpu_suffix = "-gpu" if gpu else ""

    original_algorithms = algorithms
    algorithms = flatten_if_needed(algorithms)

    plot_data = (
        data.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
    )

    pareto_data = compute_pareto(plot_data)

    qps_over_threshold = (
        pareto_data.filter(pl.col("recall") >= threshold_recall)
        .group_by("algorithm")
        .agg(pl.col("qps").max().alias("best_qps_over_thresh"))
        .sort("best_qps_over_thresh", descending=True)
    )
    legend_order = qps_over_threshold["algorithm"].to_list()
    legend_order.extend([a for a in algorithms if a not in legend_order])
    rank = {alg: i for i, alg in enumerate(legend_order)}

    tab10 = mpl.colormaps["tab10"].colors
    tab20 = mpl.colormaps["tab20"].colors
    palette = tab20 + tab10

    algorithm_colors = dict(zip(algorithms, palette))
    algorithm_dashes = dict(zip(algorithms, sns._base.unique_dashes(len(algorithms))))
    algorithm_markers = dict(zip(algorithms, sns._base.unique_markers(len(algorithms))))

    joint_legend = isinstance(original_algorithms[0], str)

    def plot_lines(pdata, background=False, ax=None):
        if ax is None:
            ax = plt.gca()
        opacity = 0.2 if background else 1.0
        kwargs = dict(data=pdata, x="recall", y="qps", alpha=opacity)
        if background:
            kwargs["color"] = "gray"
        else:
            kwargs.update(
                hue="algorithm",
                style="algorithm",
                palette=algorithm_colors,
            )
        sns.lineplot(
            units="algorithm",
            lw=1.5,
            estimator=None,
            markers=algorithm_markers,
            legend=True,
            dashes=algorithm_dashes,
            ax=ax,
            **kwargs,
        )
        ax.grid(which="major", linewidth=0.5, color="lightgray", alpha=0.5)
        ax.grid(which="minor", axis="y", linewidth=0.5, color="lightgray", alpha=0.5)

    fig, axs = plt.subplots(1, len(datasets), figsize=figsize, sharex=True, sharey=True)
    if len(datasets) == 1:
        axs = [axs]

    if joint_legend:
        algorithms = [legend_order] * len(datasets)

    for dataset, ax, algos in zip(datasets, axs, original_algorithms):
        facet = pareto_data.filter(pl.col("dataset") == dataset)
        if isinstance(algos, list):
            facet = facet.filter(pl.col("algorithm").is_in(algos))

        if facet.is_empty():
            raise ValueError(f"no results data for dataset {dataset}")

        plot_lines(facet, ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.semilogy()

        title = "-".join(dataset.split("-")[:-2])
        if dataset.endswith("-binary"):
            title = "-".join(dataset.split("-")[:-3]) + "-binary"
        ax.set_title(title)

        ticks = np.arange(xlim[0], 1.01, 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.1f}" for x in ticks])

        if pca_mahalanobis is not None:
            pca_m_data = pca_mahalanobis.filter(pl.col("dataset") == dataset)
            inset_w, inset_h, gap = 0.3, 0.3, 0.05
            bounds = [
                [0.05, 0.05, inset_w, inset_h],
                [0.05 + inset_w + gap, 0.05, inset_w, inset_h],
            ]
            axins = [ax.inset_axes(bb) for bb in bounds]
            for a in axins:
                a.spines[:].set_visible(True)
                a.spines[:].set_color("black")
                a.set_xticks([])
                a.set_yticks([])
            inset_colors = {"train": "#1f77b4", "test": "#ff7f0e"}
            for part in ["train", "test"]:
                pdata = pca_m_data.filter(pl.col("part") == part)
                axins[0].scatter(pdata["x"], pdata["y"], s=0.1, color=inset_colors[part])
            sns.kdeplot(
                pca_m_data,
                x="mahalanobis_distance_to_data",
                hue="part",
                fill=True,
                legend=False,
                ax=axins[1],
            )

    if joint_legend:
        handles, labels = axs[0].get_legend_handles_labels()

        # sort the legend entries by our rank
        sorted_pairs = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], float("inf")))
        handles, labels = zip(*sorted_pairs)

        if separate_legend:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            fig.tight_layout(pad=0.1, w_pad=1.08, h_pad=1.08)
            filename = f"{'__'.join(datasets)}-qps-recall{gpu_suffix}.png"
            print("Writing", out_dir / filename)
            fig.savefig(out_dir / filename, dpi=300)
            plt.close(fig)

            plt.figure(figsize=(1.7, 2.5))
            plt.legend(handles, labels, frameon=False)
            plt.axis("off")
            plt.tight_layout()
            filename = f"{'__'.join(datasets)}-qps-recall{gpu_suffix}-legend.png"
            print("Writing", out_dir / filename)
            plt.savefig(out_dir / filename, dpi=300)
            plt.close()

        else:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            legend_pad = 0.20
            fig.tight_layout(rect=[0, 0, 1 - legend_pad, 1], pad=0.1, w_pad=1.08, h_pad=1.08)
            fig.legend(handles, labels, loc="center right")
            filename = f"{'__'.join(datasets)}-qps-recall{gpu_suffix}.png"
            print("Writing", out_dir / filename)
            fig.savefig(out_dir / filename, dpi=300)
            plt.close(fig)

    else:
        for ax in axs:
            if ax.get_legend() is not None:
                handles, labels = ax.get_legend_handles_labels()
                sorted_pairs = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], float("inf")))
                handles, labels = zip(*sorted_pairs)
                ax.legend(handles, labels)

        fig.tight_layout(pad=0.1, w_pad=1.08, h_pad=1.08)
        filename = f"{'__'.join(datasets)}-qps-recall{gpu_suffix}.png"
        print("Writing", out_dir / filename)
        fig.savefig(out_dir / filename, dpi=300)
        plt.close(fig)


def split_difficulties_plot(
    out_dir,
    summary,
    detail,
    query_stats,
    recall,
    datasets,
    algorithms=["symphonyqg", "lorann", "glass", "ngt-qg"],
    k=100,
    easy_ptile=0.1,
    difficult_ptile=0.1,
    gpu=False,
):
    nqueries = query_stats.group_by("dataset").len("nqueries")
    gpu_suffix = "-gpu" if gpu else ""

    actual_performance_data = (
        # Pick the relevant data
        summary.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
        # Compute the throughput and recall of each algorithm configuration
        # Select the fastest configuration with recall above the threshold,
        # for each algorithm and difficulty
        .filter(pl.col("recall") > recall)
        .with_columns(pl.col("qps").rank(descending=True).over(["dataset", "algorithm"]).alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("dataset", "algorithm", "params", "qps", "recall")
    )

    selected_queries = (
        query_stats.select("dataset", "query_index", "rc100")
        .with_columns(pl.col("rc100").rank("ordinal", descending=True).over("dataset").alias("rank_rc100"))
        .join(nqueries, on="dataset")
        .with_columns(
            pl.when(pl.col("rank_rc100") < easy_ptile * pl.col("nqueries"))
            .then(pl.lit("easy"))
            .when(pl.col("rank_rc100") >= (1 - difficult_ptile) * pl.col("nqueries"))
            .then(pl.lit("difficult"))
            .alias("difficulty")
        )
        .drop_nulls("difficulty")
    )

    plot_data = (
        # Pick the relevant data
        detail.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
        .join(actual_performance_data, on=["dataset", "algorithm", "params"], how="semi")
        # Select only the easy and difficult queries, for all algorithm's parameterizations
        .join(selected_queries, on=["dataset", "query_index"])
        .select(pl.exclude("k", "query_index", "rank_rc100", "nqueries"))
        # Compute the throughput and recall of each algorithm configuration
        # on these "virtual" workloads
        .group_by("dataset", "algorithm", "params", "difficulty")
        .agg(pl.col("recall").mean(), (1 / pl.col("time").mean()).alias("qps"))
        .select("dataset", "algorithm", "difficulty", "qps", "recall")
    )

    plot_data = plot_data.pivot(index=["dataset", "algorithm"], on="difficulty", values="recall").sort("difficult")

    height = 0.26 * plot_data.select("algorithm").n_unique()
    _, axs = plt.subplots(1, len(datasets), figsize=(8, height))
    if len(datasets) == 1:
        axs = [axs]

    def do_plot(pdata, ax):
        ax.hlines(range(pdata.shape[0]), xmin=pdata["easy"], xmax=pdata["difficult"], color="grey", alpha=0.4)
        ax.scatter(pdata["difficult"], pdata["algorithm"], zorder=2, clip_on=False, color="tab:blue", label="difficult")
        ax.scatter(pdata["easy"], pdata["algorithm"], zorder=2, clip_on=False, color="#c9a227", label="easy")

        ax.axvline(recall, color="lightgray", lw=1, zorder=-1)

        for algo in pdata["algorithm"].unique().to_list():
            xpos = pdata.filter(pl.col("algorithm") == algo)[["difficult", "easy"]].transpose().min()["column_0"][0]
            performance_easy, performance_difficult = tuple(
                pdata.filter(pl.col("algorithm") == algo)[["easy", "difficult"]].unpivot()["value"]
            )
            if xpos is not None:
                ax.annotate(
                    xy=(xpos, algo),
                    xytext=(-35, 0),
                    textcoords="offset points",
                    text=algo,
                    ha="right",
                    va="center",
                    size=9,
                )
            if performance_difficult is not None:
                ax.annotate(
                    xy=(performance_difficult, algo),
                    text=f"{performance_difficult:.2f}",
                    ha="right" if performance_easy > performance_difficult else "left",
                    va="center",
                    size=9,
                    color="steelblue",
                    xytext=(-10, 0) if performance_easy > performance_difficult else (10, 0),
                    textcoords="offset points",
                )
            if performance_easy is not None:
                ax.annotate(
                    xy=(performance_easy, algo),
                    text=f"{performance_easy:.2f}",
                    ha="left" if performance_easy > performance_difficult else "right",
                    va="center",
                    size=9,
                    color="#c9a227",
                    xytext=(10, 0) if performance_easy > performance_difficult else (-10, 0),
                    textcoords="offset points",
                )
        ax.axis("off")

    for dataset, ax in zip(datasets, axs):
        do_plot(plot_data.filter(pl.col("dataset") == dataset), ax)
        ax.set_title(dataset, pad=15)

    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    filename = f"split-performance-{'__'.join(datasets)}{gpu_suffix}.png"
    print("Writing", out_dir / filename)
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def dataset_geometry_grid(out_dir, pca_mahalanobis, datasets=None, n_cols=3, max_scatter=2000):
    """Grid PDF: PCA scatter + Mahalanobis KDE side by side for every dataset."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch

    if datasets is None:
        # Use all datasets present in the parquet, ordered by known groups then alphabetically
        known_order = ID_DATASETS + ID_DATASETS_ADDITIONAL + OOD_DATASETS
        available = set(pca_mahalanobis["dataset"].unique().to_list())
        datasets = [d for d in known_order if d in available] + sorted(
            available - set(known_order)
        )

    datasets = [d for d in datasets if d in set(pca_mahalanobis["dataset"].unique().to_list())]

    n = len(datasets)
    n_rows = math.ceil(n / n_cols)
    colors = {"train": "#1f77b4", "test": "#ff7f0e"}

    fig = plt.figure(figsize=(n_cols * 4.0, n_rows * 2.5))
    outer = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    for i, dataset in enumerate(datasets):
        row, col = divmod(i, n_cols)
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.1)
        ax_pca = fig.add_subplot(inner[0])
        ax_kde = fig.add_subplot(inner[1])

        pdata = pca_mahalanobis.filter(pl.col("dataset") == dataset)

        parts = dataset.split("-")
        if dataset.endswith("-binary"):
            title = "-".join(parts[:-3]) + "-binary"
        elif len(parts) >= 3:
            title = "-".join(parts[:-2])
        else:
            title = dataset

        for part in ["train", "test"]:
            p = pdata.filter(pl.col("part") == part)
            if p.height > max_scatter:
                p = p.sample(max_scatter, seed=42)
            ax_pca.scatter(
                p["x"].to_numpy(), p["y"].to_numpy(),
                s=0.3, alpha=0.6, color=colors[part], rasterized=True,
            )
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        title_color = "red" if dataset in NEW_DATASETS else "black"
        ax_pca.set_title(title, fontsize=7, pad=2, color=title_color)

        sns.kdeplot(
            pdata.to_pandas(),
            x="mahalanobis_distance_to_data",
            hue="part",
            hue_order=["train", "test"],
            palette=colors,
            fill=True,
            common_norm=False,
            legend=False,
            ax=ax_kde,
        )
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.set_xlabel("")
        ax_kde.set_ylabel("")

    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.1)
        for j in range(2):
            fig.add_subplot(inner[j]).axis("off")

    legend_handles = [Patch(color=colors["train"], label="data"), Patch(color=colors["test"], label="queries")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0), fontsize=8)

    filename = out_dir / "dataset-geometry-grid.pdf"
    print("Writing", filename)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_ridgeline(out_dir, query_stats, x="rc100", log=True):
    # adapted from https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    from sklearn.neighbors import KernelDensity
    import numpy as np

    nan_counts = (
        query_stats
        .filter(~pl.col("dataset").str.contains("-ip"))
        .filter(~pl.col("dataset").str.contains("-dot"))
        .group_by("dataset")
        .agg(pl.col(x).is_nan().sum().alias("nan_count"))
        .filter(pl.col("nan_count") > 0)
        .sort("dataset")
    )
    if nan_counts.is_empty():
        print(f"No NaN values in {x}")
    else:
        print(f"NaN rows in {x} per dataset:")
        for row in nan_counts.iter_rows(named=True):
            print(f"  {row['dataset']}: {row['nan_count']}")

    query_stats = (
        query_stats
        .filter(~pl.col("dataset").str.contains("-ip"))
        .filter(~pl.col("dataset").str.contains("-dot"))
        .filter(pl.col(x).is_not_nan())
        .with_columns(mean_x=pl.col(x).mean().over("dataset"))
        .with_columns(
            pl.when(pl.col("dataset").is_in(ID_DATASETS))
            .then(pl.lit("in-distribution"))
            .when(pl.col("dataset").is_in(OOD_DATASETS))
            .then(pl.lit("out-of-distribution"))
            .otherwise(pl.lit("ann-benchmarks"))
            .alias("dataset-type"),
        )
        .sort("mean_x", descending=True)
    )

    if log:
        query_stats = query_stats.with_columns(pl.col(x).log())

    datasets = query_stats.group_by("dataset").agg(pl.col(x).median()).sort(x, descending=True)["dataset"].to_list()

    fig_height = max(3, len(datasets) * 0.5)
    plt.figure(figsize=(8, fig_height))
    ax = plt.gca()

    maxx = max(3.5, math.ceil(query_stats[x].max()))
    minx = min(0, math.floor(query_stats[x].min()))
    for i, dataset in enumerate(datasets):
        pdata = query_stats.filter(pl.col("dataset") == dataset)
        xvals = pdata[x].drop_nans().to_numpy()
        if len(xvals) == 0:
            continue
        x_d = np.linspace(minx, maxx, 1000)

        kde = KernelDensity(bandwidth=0.05, kernel="gaussian")
        kde.fit(xvals[:, None])
        logprob = kde.score_samples(x_d[:, None])

        offset = (len(datasets) - i - 1) * 1.5
        if dataset in ID_DATASETS:
            color = "tab:blue"
        elif dataset in OOD_DATASETS:
            color = "tab:orange"
        else:
            color = "tab:green"
        ax.plot(x_d, offset + np.exp(logprob), color="#f0f0f0", lw=1, zorder=2 * i + 1)
        ax.fill_between(x_d, offset + np.exp(logprob), offset, alpha=1, zorder=2 * i, color=color)
        parts = dataset.split("-")
        label = dataset if len(parts) < 3 else "-".join(parts[:-2])
        ax.annotate(label, (maxx, offset), ha="right", va="bottom", color=color)

    if log:
        ax.set_xlabel(f"log({x})")
    else:
        ax.set_xlabel(f"{x}")

    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    filename = out_dir / f"distribution-{x}.png"
    print("Writing", filename)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_filtered_rc_ridgeline(
    out_dir,
    query_stats,
    data_dir=None,
    wiki_src=None,
    k=100,
    log=True,
    yfcc_n_queries=50,
):
    """
    Plot RC evolution across selectivity levels for filtered workloads.

    RC_filtered = dMean / dK_filtered where dMean (average distance to all
    training vectors) is fixed per query and dK_filtered is recomputed via
    brute-force against the filtered subset at each selectivity level.

    As the filter tightens (fewer vectors pass), dK_filtered grows and
    RC_filtered falls → curves shift left (harder queries).

    Panels:
      arxiv_1M          – 50 queries, selectivity levels where threshold < 1M
      wiki_1M uncorr.   – 50 queries, selectivity levels where all 1M chunk_ids
                           are below the threshold
      yfcc_1M           – yfcc_n_queries sampled queries, one natural filter each
    """
    import h5py
    from sklearn.neighbors import KernelDensity

    if data_dir is None:
        data_dir = pathlib.Path("data")

    # ------------------------------------------------------------------
    # Brute-force streaming K-NN helper
    # ------------------------------------------------------------------
    def _brute_force_dk(query_vecs, train_ds, filter_mask, k_nb, chunk_size=50_000):
        """
        Compute the k_nb-th nearest distance for each query against the
        filtered subset of train_ds.  Streams train_ds in chunks to bound
        memory.  Returns shape (n_queries,); NaN when fewer than k_nb
        filtered vectors exist.
        """
        n_total = len(filter_mask)
        n_filtered = int(filter_mask.sum())
        if n_filtered < k_nb:
            return np.full(len(query_vecs), np.nan, dtype=np.float32)

        q = query_vecs.astype(np.float32)
        q_sq = (q * q).sum(axis=1, keepdims=True)            # (n_q, 1)
        top = np.full((len(q), k_nb), np.inf, dtype=np.float32)

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            local_mask = filter_mask[start:end]
            if not local_mask.any():
                continue
            chunk = train_ds[start:end][local_mask].astype(np.float32)
            c_sq = (chunk * chunk).sum(axis=1)
            dot = q @ chunk.T
            sq_d = q_sq + c_sq[None, :] - 2.0 * dot
            d = np.sqrt(np.maximum(sq_d, 0.0))
            combined = np.concatenate([top, d], axis=1)
            top = np.partition(combined, k_nb - 1, axis=1)[:, :k_nb]

        return np.sort(top, axis=1)[:, k_nb - 1]

    def _draw_ridgeline(ax, records, sel_order, sel_labels, x_d, log_scale, cmap):
        n = len(sel_order)
        for idx, (key, label) in enumerate(zip(sel_order, sel_labels)):
            vals = records[key]
            vals = vals[~np.isnan(vals)]
            if log_scale:
                vals = np.log(vals[vals > 0])
            if len(vals) < 2:
                continue
            kde = KernelDensity(bandwidth=0.05, kernel="gaussian")
            kde.fit(vals[:, None])
            density = np.exp(kde.score_samples(x_d[:, None]))
            offset = (n - idx - 1) * 1.2
            color = cmap(idx / max(n - 1, 1))
            ax.plot(x_d, offset + density, color="#f0f0f0", lw=0.8, zorder=2 * idx + 1)
            ax.fill_between(x_d, offset + density, offset, alpha=0.85,
                            zorder=2 * idx, color=color)
            ax.annotate(label, (x_d[-1], offset), ha="right", va="bottom",
                        color=color, fontsize=8)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    # ------------------------------------------------------------------
    # arxiv_1M
    # ------------------------------------------------------------------
    UNIFORM_SELS = [1, 3, 5, 10, 20, 30, 40, 50, 75, 90]  # % of 1M train; 100% = unfiltered baseline

    arxiv_records = {}
    arxiv_sel_order = []
    arxiv_sel_labels = []
    arxiv_unfiltered = None
    try:
        stats_a = (
            query_stats.filter(pl.col("dataset") == "arxiv_1M").sort("query_index")
        )
        with h5py.File(data_dir / "arxiv_1M.hdf5", "r") as f:
            n_train = f["train"].shape[0]
            test_vecs = f["test"][:]
            d_unfiltered = f["distances"][:, k - 1]
            dMean = stats_a["rc100"].to_numpy() * d_unfiltered
            arxiv_unfiltered = stats_a["rc100"].to_numpy()   # baseline curve

            for pct in UNIFORM_SELS:
                threshold = round(pct / 100 * n_train)
                if threshold < k or threshold >= n_train:
                    continue
                filter_mask = np.arange(n_train) < threshold
                dk = _brute_force_dk(test_vecs, f["train"], filter_mask, k)
                arxiv_records[pct] = dMean / np.where(dk > 0, dk, np.nan)
                arxiv_sel_order.append(pct)
                arxiv_sel_labels.append(f"{pct}%")

        # append unfiltered baseline at the end (easiest → bottom)
        arxiv_records["unfiltered"] = arxiv_unfiltered
        arxiv_sel_order.append("unfiltered")
        arxiv_sel_labels.append("unfiltered")
        print(f"arxiv_1M: {len(arxiv_sel_order)} levels (incl. unfiltered baseline)")
    except Exception as exc:
        print(f"Skipping arxiv_1M filtered RC: {exc}")
        arxiv_records.clear(); arxiv_sel_order.clear(); arxiv_sel_labels.clear()

    # ------------------------------------------------------------------
    # wiki_1M uncorrelated – brute-force on 1M subset
    # ------------------------------------------------------------------
    wiki_records = {}
    wiki_sel_order = []
    wiki_sel_labels = []
    wiki_unfiltered = None
    try:
        stats_w = (
            query_stats
            .filter(pl.col("dataset") == "wiki_1M")
            .filter(pl.col("query_index") < 50)
            .sort("query_index")
        )
        bm = pl.read_parquet(data_dir / "wiki_1M_base_metadata.parquet")
        chunk_ids = bm["chunk_id"].to_numpy()

        with h5py.File(data_dir / "wiki_1M.hdf5", "r") as f:
            n_train_w = f["train"].shape[0]
            test_vecs = f["test"][:50]
            d_unfiltered = f["distances"][:50, k - 1]
            dMean = stats_w["rc100"].to_numpy() * d_unfiltered
            wiki_unfiltered = stats_w["rc100"].to_numpy()

            for pct in UNIFORM_SELS:
                threshold = round(pct / 100 * n_train_w)
                filter_mask = chunk_ids < threshold
                if filter_mask.sum() < k:
                    continue
                dk = _brute_force_dk(test_vecs, f["train"], filter_mask, k)
                wiki_records[pct] = dMean / np.where(dk > 0, dk, np.nan)
                wiki_sel_order.append(pct)
                wiki_sel_labels.append(f"{pct}%")

        wiki_records["unfiltered"] = wiki_unfiltered
        wiki_sel_order.append("unfiltered")
        wiki_sel_labels.append("unfiltered")
        print(f"wiki_1M uncorr: {len(wiki_sel_order)} levels (incl. unfiltered baseline)")
    except Exception as exc:
        print(f"Skipping wiki_1M filtered RC: {exc}")
        wiki_records.clear(); wiki_sel_order.clear(); wiki_sel_labels.clear()

    # ------------------------------------------------------------------
    # yfcc_1M – per-query tag-intersection filter, sampled queries
    # ------------------------------------------------------------------
    yfcc_rc = None
    try:
        stats_y = (
            query_stats.filter(pl.col("dataset") == "yfcc_1M").sort("query_index")
        )
        ym = pl.read_parquet(data_dir / "yfcc_1M_query_metadata.parquet")
        yb = pl.read_parquet(data_dir / "yfcc_1M_base_metadata.parquet")

        # Build tag inverted index: tag_id → sorted array of vector_ids
        tmp: dict = {}
        for vid, tags in enumerate(yb["tag_ids"].to_list()):
            for tag in tags:
                if tag not in tmp:
                    tmp[tag] = []
                tmp[tag].append(vid)
        tag_inv = {t: np.array(v, dtype=np.int32) for t, v in tmp.items()}
        del tmp

        # Sample queries deterministically (first yfcc_n_queries by query_id)
        sample = ym.sort("query_id").head(yfcc_n_queries)
        n_train = len(yb)

        with h5py.File(data_dir / "yfcc_1M.hdf5", "r") as f:
            rc100_y = stats_y.head(yfcc_n_queries)["rc100"].to_numpy()
            d_unfilt = f["distances"][:yfcc_n_queries, k - 1]
            dMean_y  = rc100_y * d_unfilt
            test_vecs = f["test"][:yfcc_n_queries]

            dk_all = np.full(yfcc_n_queries, np.nan, dtype=np.float32)
            for i, row in enumerate(sample.iter_rows(named=True)):
                tags = row["tag_ids"] or []
                parts = [tag_inv[t] for t in tags if t in tag_inv]
                if not parts:
                    continue
                matching = np.unique(np.concatenate(parts))
                filter_mask = np.zeros(n_train, dtype=bool)
                filter_mask[matching] = True
                dk = _brute_force_dk(test_vecs[i:i+1], f["train"], filter_mask, k)
                dk_all[i] = dk[0]

        yfcc_rc = dMean_y / np.where(dk_all > 0, dk_all, np.nan)
        yfcc_rc_unfiltered = rc100_y
        print(f"yfcc_1M: computed filtered RC for {(~np.isnan(yfcc_rc)).sum()} / {yfcc_n_queries} queries")
    except Exception as exc:
        print(f"Skipping yfcc_1M filtered RC: {exc}")

    if not arxiv_records and not wiki_records and yfcc_rc is None:
        print("No filtered RC data; skipping plot.")
        return

    # ------------------------------------------------------------------
    # Plot – one panel per dataset
    # ------------------------------------------------------------------
    panels = []
    if arxiv_records:
        panels.append(("arxiv_1M", arxiv_records, arxiv_sel_order, arxiv_sel_labels))
    if wiki_records:
        panels.append(("wiki_1M (uncorr.)", wiki_records, wiki_sel_order, wiki_sel_labels))
    if yfcc_rc is not None:
        # Pack as single-level records dict for reuse of _draw_ridgeline
        panels.append(("yfcc_1M", {"filtered": yfcc_rc, "unfiltered": yfcc_rc_unfiltered},
                       ["filtered", "unfiltered"], ["tag-filtered", "unfiltered"]))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)

    for ax, (title, records, sel_order, sel_labels) in zip(axes[0], panels):
        all_vals = np.concatenate([
            v[~np.isnan(v)] for v in records.values() if hasattr(v, "__len__")
        ])
        if log:
            all_vals = np.log(all_vals[all_vals > 0])
        x_min = min(-0.5, math.floor(all_vals.min() * 2) / 2)
        x_max = max(3.5, math.ceil(all_vals.max() * 2) / 2)
        x_d = np.linspace(x_min, x_max, 1000)

        n = len(sel_order)
        cmap = mpl.colormaps.get_cmap("coolwarm_r").resampled(n)
        _draw_ridgeline(ax, records, sel_order, sel_labels, x_d, log, cmap)
        ax.set_xlabel("log(RC)" if log else "RC")
        ax.set_title(title)

    fig.suptitle(f"Filtered RC (k={k}): curves shift left as filter tightens → harder queries")
    plt.tight_layout()
    filename = out_dir / "distribution-filtered-rc.png"
    print("Writing", filename)
    plt.savefig(filename, dpi=300)
    plt.close()


def performance_gap_plot(out_dir, id_dataset, ood_dataset, summary, pca_mahalanobis_data, k=100, recall=0.9, gpu=False):
    """Plot the performance difference between in-distribution and out of distribution queries"""
    from matplotlib.gridspec import GridSpec

    gpu_suffix = "-gpu" if gpu else ""
    pdata = summary.filter(pl.col("dataset").is_in([id_dataset, ood_dataset])).filter(pl.col("k") == k)
    if pdata.is_empty():
        raise ValueError("no results data found for performance gap plot")

    maindata = (
        fastest_at(pdata, recall)
        .with_columns(
            pl.when(pl.col("dataset") == id_dataset)
            .then(pl.lit("in-distribution"))
            .otherwise(pl.lit("out-of-distribution"))
            .alias("type")
        )
        .pivot(on="type", values="qps", index="algorithm")
        .drop_nulls()
        .sort("out-of-distribution", descending=False)
    )

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(2, 3)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_pca_id = fig.add_subplot(gs[0, 1])
    ax_pca_ood = fig.add_subplot(gs[1, 1])
    ax_mahalanobis_id = fig.add_subplot(gs[0, 2])
    ax_mahalanobis_ood = fig.add_subplot(gs[1, 2])
    ax_main.hlines(
        range(maindata.shape[0]),
        xmin=maindata["out-of-distribution"],
        xmax=maindata["in-distribution"],
        color="gray",
        zorder=-1,
    )
    ax_main.scatter(maindata["in-distribution"], maindata["algorithm"], color="tab:green")
    ax_main.scatter(maindata["out-of-distribution"], maindata["algorithm"], color="tab:purple")

    for algo in maindata["algorithm"].unique().to_list():
        xpos = (
            maindata.filter(pl.col("algorithm") == algo)[["in-distribution", "out-of-distribution"]]
            .transpose()
            .min()["column_0"][0]
        )
        performance_id, performance_ood = tuple(
            maindata.filter(pl.col("algorithm") == algo)[["in-distribution", "out-of-distribution"]].unpivot()["value"]
        )
        if xpos is not None:
            ax_main.annotate(
                xy=(xpos, algo), xytext=(-35, 0), textcoords="offset points", text=algo, ha="right", va="center", size=9
            )
        if performance_ood is not None:
            ax_main.annotate(
                xy=(performance_ood, algo),
                text=f"{performance_ood:.0f}",
                ha="right" if performance_id > performance_ood else "left",
                va="center",
                size=9,
                color="tab:purple",
                xytext=(-8, 0) if performance_id > performance_ood else (10, 0),
                textcoords="offset points",
            )
        if performance_id is not None:
            ax_main.annotate(
                xy=(performance_id, algo),
                text=f"{performance_id:.0f}",
                ha="left" if performance_id > performance_ood else "right",
                va="center",
                size=9,
                color="tab:green",
                xytext=(8, 0) if performance_id > performance_ood else (-10, 0),
                textcoords="offset points",
            )
    ax_main.axis("off")

    for dataname, ax_pca, ax_mahalanobis, color in zip(
        [id_dataset, ood_dataset],
        [ax_pca_id, ax_pca_ood],
        [ax_mahalanobis_id, ax_mahalanobis_ood],
        ["tab:green", "tab:purple"],
    ):
        pdata = pca_mahalanobis_data.filter(pl.col("dataset") == dataname, pl.col("part") == "train")
        ax_pca.scatter(pdata["x"], pdata["y"], s=0.1, c="tab:blue")
        sns.kdeplot(
            pdata, x="mahalanobis_distance_to_data", color="tab:blue", fill=True, legend=False, ax=ax_mahalanobis
        )

        pdata = pca_mahalanobis_data.filter(pl.col("dataset") == dataname, pl.col("part") == "test")
        ax_pca.scatter(pdata["x"], pdata["y"], s=0.1, c=color)
        sns.kdeplot(pdata, x="mahalanobis_distance_to_data", color=color, fill=True, legend=False, ax=ax_mahalanobis)
        ax_pca.axis("off")
        ax_mahalanobis.set_yticks([])
        ax_mahalanobis.set_xticks([])
        ax_mahalanobis.set_xlabel("")
        ax_mahalanobis.set_ylabel("")
        ax_mahalanobis.spines[:].set_visible(False)
        ax_mahalanobis.spines["bottom"].set_visible(True)

    plt.tight_layout()
    filename = f"performance-gap-{ood_dataset}{gpu_suffix}.png"
    print("Writing", out_dir / filename)
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()


def paper(out_dir, all_algorithms, summary, detail, query_stats, pca_mahalanobis):
    selected = [
        "glass",
        "hnswlib",
        "ivfpqfs(faiss)",
        "lorann",
        "ngt-onng",
        "ngt-qg",
        "pynndescent",
        "scann",
        "symphonyqg",
    ]

    plot_difficulty_ridgeline(out_dir, query_stats)

    radar_at_recall_plot(
        out_dir,
        summary,
        query_stats,
        algorithms=selected,
        all_algorithms=all_algorithms,
        recall=0.95,
    )

    for dataset in ["agnews-mxbai-1024-euclidean", "arxiv-nomic-768-normalized"]:
        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=None,
            datasets=[dataset],
            algorithms=all_algorithms,
            xlim=(0.7, 1.0),
            ylim=(1e1, 1.1e4),
            separate_legend=False,
            gpu=False,
        )

    for datasets in [
        ["imagenet-clip-512-normalized", "landmark-nomic-768-normalized"],
        ["agnews-mxbai-1024-euclidean", "arxiv-nomic-768-normalized"],
        ["ccnews-nomic-768-normalized", "celeba-resnet-2048-cosine"],
        ["codesearchnet-jina-768-cosine", "glove-200-cosine"],
        ["gooaq-distilroberta-768-normalized", "imagenet-clip-512-normalized"],
        ["landmark-dino-768-cosine", "landmark-nomic-768-normalized"],
        ["simplewiki-openai-3072-normalized", "yahoo-minilm-384-normalized"],
    ]:
        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=None,
            datasets=datasets,
            algorithms=selected,
            xlim=(0.7, 1.0),
            ylim=(1e2, 1.1e4),
            figsize=(8, 3),
            separate_legend=True,
        )

    for datasets in [
        ["coco-nomic-768-normalized", "imagenet-align-640-normalized"],
        ["laion-clip-512-normalized", "yandex-200-cosine"],
    ]:
        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=pca_mahalanobis,
            datasets=datasets,
            algorithms=[
                "glass",
                "hnswlib",
                "ivfpqfs(faiss)",
                "lorann",
                "mlann-rf",
                "ngt-qg",
                "roargraph",
                "scann",
                "symphonyqg",
            ],
            xlim=(0.5, 1),
            ylim=(1e2, 1.1e4),
            figsize=(8, 3),
            separate_legend=True,
        )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=pca_mahalanobis,
        datasets=["yi-128-ip", "llama-128-ip"],
        algorithms=[
            "glass",
            "hnswlib",
            "ivf(faiss)",
            "ivfpqfs(faiss)",
            "lorann",
            "mlann-rf",
            "ngt-onng",
            "roargraph",
            "scann",
        ],
        xlim=(0, 1),
        ylim=(1e2, 3e4),
        figsize=(8, 3),
        separate_legend=True,
    )

    split_difficulties_plot(
        out_dir,
        summary,
        detail,
        query_stats,
        0.90,
        datasets=["arxiv-nomic-768-normalized", "landmark-nomic-768-normalized"],
    )

    for datasets in [
        ["agnews-mxbai-1024-hamming-binary", "agnews-mxbai-1024-euclidean"],
        ["ccnews-nomic-768-hamming-binary", "ccnews-nomic-768-normalized"],
        ["landmark-nomic-768-hamming-binary", "landmark-nomic-768-normalized"],
        ["simplewiki-openai-3072-hamming-binary", "simplewiki-openai-3072-normalized"],
    ]:
        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=None,
            datasets=datasets,
            algorithms=[
                ["ngt-onng", "ivf(faiss)", "pynndescent", "hnsw(faiss)"],
                ["cuvs-cagra", "cuvs-ivfpq", "faiss-gpu-ivf", "cuvs-ivf", "ggnn"],
            ],
            xlim=(0.7, 1),
            ylim=(3e2, 3e5),
            figsize=(8, 3),
            separate_legend=False,
            gpu=True,
        )

    for datasets in [
        ("coco-nomic-id-768-normalized", "coco-nomic-768-normalized"),
        ("imagenet-align-id-640-normalized", "imagenet-align-640-normalized"),
        ("laion-clip-id-512-normalized", "laion-clip-512-normalized"),
        ("yandex-id-200-cosine", "yandex-200-cosine"),
    ]:
        performance_gap_plot(
            out_dir,
            datasets[0],
            datasets[1],
            summary,
            pca_mahalanobis,
            recall=0.95,
            gpu=False,
        )

    latency_difference_plot(
        summary,
        detail,
        0.95,
        ID_DATASETS + ID_DATASETS_ADDITIONAL + OOD_DATASETS,
        output=out_dir,
        algorithms=selected,
        significance_level=0.01,
        gpu=False,
    )


def latency_difference_table(
    data,
    detail,
    algorithms,
    recall,
    dataset,
    k=100,
):
    """\
    Checks whether the differences in latencies are statistically significant or not,
    by performing the Wilcoxon paired test.
    """
    data = data.filter(pl.col("dataset") == dataset).filter(pl.col("algorithm").is_in(algorithms))

    configs = (
        fastest_at(data, recall, k).select("dataset", "algorithm", "params").sort("dataset", "algorithm", "params")
    )

    stats = detail.filter(pl.col("k") == k, pl.col("dataset") == dataset).join(
        configs, on=["dataset", "algorithm", "params"]
    )

    def get_times(stats, algorithm):
        times = (
            stats.filter(pl.col("algorithm") == algorithm)
            .group_by("query_index")
            .agg(pl.col("time").mean())
            .sort("query_index")
        )["time"]
        return times.to_numpy()

    p_values = []
    for a, b in itertools.combinations(algorithms, 2):
        atimes = get_times(stats, a)
        btimes = get_times(stats, b)
        if len(atimes) == 0 or len(btimes) == 0:
            continue
        # Do the pairwise test
        test = wilcoxon(atimes - btimes)
        p_values.append(
            dict(algorithm_a=a, algorithm_b=b, latency_a=atimes.mean(), latency_b=btimes.mean(), p_value=test.pvalue)
        )
    if len(p_values) == 0:
        return pl.DataFrame(schema=["algorithm_a", "algorithm_b", "latency_a", "latency_b", "p_value", "dataset"])
    return pl.DataFrame(p_values).sort("p_value").with_columns(pl.lit(dataset).alias("dataset"))


def holm_bonferroni(table, p_value_col):
    """Correct the p-values of the given table, where each row is a statistical test."""
    table = table.sort(p_value_col)
    sorted_p_values = table[p_value_col].to_numpy()
    corrected_p_values = np.maximum.accumulate(sorted_p_values * np.arange(len(sorted_p_values), 0, -1))
    corrected_p_values = np.minimum(corrected_p_values, 1)
    table = table.with_columns(pl.Series(name=p_value_col, values=corrected_p_values))
    return table


def latency_difference_plot(
    summary, detail, recall, datasets, algorithms, output, k=100, significance_level=0.05, gpu=False
):
    try:
        import networkx
    except ImportError:
        raise ImportError("latency_difference_plot requires networkx")

    gpu_suffix = "-gpu" if gpu else ""
    tests = []
    for dataset in datasets:
        df = latency_difference_table(summary, detail, algorithms, recall, dataset, k=k)
        tests.append(df)
    tests = holm_bonferroni(pl.concat(tests), "p_value")
    print(
        tests.filter(pl.col("p_value") < 0.01).shape[0], "tests out of", tests.shape[0], "are statistically significant"
    )

    graphs = dict()
    for dataset in datasets:
        G = networkx.Graph()
        for algo in algorithms:
            G.add_node(algo)
        graphs[dataset] = G
    for test in tests.rows(named=True):
        if (
            test["p_value"] >= significance_level
            and test["algorithm_a"] in algorithms
            and test["algorithm_b"] in algorithms
        ):
            graphs[test["dataset"]].add_edge(test["algorithm_a"], test["algorithm_b"])

    for dataset in datasets:
        G = graphs[dataset]
        groups = [c for c in networkx.find_cliques(G) if len(c) > 1]
        plt.figure(figsize=(8, 2))
        pdata = tests.filter(pl.col("dataset") == dataset)
        pdata = (
            pl.concat(
                [
                    pdata.select(algorithm="algorithm_a", latency="latency_a"),
                    pdata.select(algorithm="algorithm_b", latency="latency_b"),
                ]
            )
            .unique()
            .sort("latency")
        )

        algos = pdata["algorithm"].to_numpy()
        times = pdata["latency"].to_numpy()
        minx, maxx = times[0], times[-1]
        span = maxx - minx
        for x in [minx, maxx]:
            label = f"{x * 1000:.3} ms"
            plt.annotate(label, xy=(x, 0), xytext=(x, 0.15), ha="center")
            plt.plot((x, x), (0, 0.05), c="black", lw=0.5)
        plt.plot((minx, maxx), (0, 0), c="black")
        minx = minx - 0.1 * span
        maxx = maxx + 0.1 * span

        offset = 0.2
        baseline = -0.3
        for i, a in enumerate(algos):
            if i < len(algos) // 2:
                y = baseline - i * offset
                plt.annotate(a, xy=(minx - 0.02 * span, y), ha="right")
                plt.plot((minx, times[i], times[i]), (y, y, 0), lw=0.5, c="black")
            else:
                y = baseline - (len(algos) - i - 1) * offset
                plt.annotate(a, xy=(maxx + 0.02 * span, y), ha="left")
                plt.plot((maxx, times[i], times[i]), (y, y, 0), lw=0.5, c="black")

        offset = 0.1
        baseline = -0.1
        for i, group in enumerate(groups):
            gstart = pdata.filter(pl.col("algorithm").is_in(group))["latency"].min()
            gend = pdata.filter(pl.col("algorithm").is_in(group))["latency"].max()
            y = baseline - i * offset
            plt.plot((gstart - 0.005 * span, gend + 0.005 * span), (y, y), lw=3, c="black")

        plt.title(dataset)
        plt.gca().set_axis_off()
        plt.tight_layout()
        filename = f"latency-critdiff-{dataset}-{recall}{gpu_suffix}.png"
        print("Writing", pathlib.Path(output) / filename)
        plt.savefig(pathlib.Path(output) / filename, dpi=600)


def print_metric_table(data, recall=0.9, k=100, algorithms=None, metric="build_time"):
    filtered_datasets = [
        "agnews-mxbai-1024-euclidean",
        "arxiv-nomic-768-normalized",
        "gooaq-distilroberta-768-normalized",
        "imagenet-clip-512-normalized",
        "landmark-nomic-768-normalized",
        "yahoo-minilm-384-normalized",
    ]

    filtered_data = data.filter(pl.col("k") == k).filter(pl.col("dataset").is_in(filtered_datasets))
    if algorithms is not None:
        filtered_data = filtered_data.filter(pl.col("algorithm").is_in(algorithms))

    best_qps_points = (
        filtered_data.filter(pl.col("recall") >= recall)
        .sort("qps", descending=True)
        .group_by(["dataset", "algorithm"])
        .first()
        .select(["dataset", "algorithm", "recall", "build_time", "index_size", "qps"])
        .sort(["dataset", "algorithm"])
    )

    if best_qps_points.is_empty():
        print(f"No points found exceeding recall {recall}")
        return

    pivot_table = best_qps_points.pivot(on="dataset", index="algorithm", values=metric)

    datasets = [col for col in pivot_table.columns if col != "algorithm"]
    pivot_table = pivot_table.with_columns(
        pl.concat_list([pl.col(ds) for ds in datasets]).list.mean().alias("avg_metric")
    ).sort("avg_metric")

    datasets = [col for col in pivot_table.columns if col not in ["algorithm", "avg_metric"]]
    dataset_short_names = [ds.split("-")[0] for ds in datasets]

    print("\\begin{table}[ht!]")
    print("\\begin{center}")
    metric_description = "Index construction times (seconds)" if metric == "build_time" else "Index sizes (KB)"
    sort_description = "index construction times" if metric == "build_time" else "index sizes"
    print(
        f"\\caption{{{metric_description} with throughput-optimized hyperparameters at ${int(recall * 100)}\\%$ recall. "
        f"Algorithms are sorted based on their average {sort_description}.}}"
    )
    print("\\label{table:construction}")
    print("\\begin{NiceTabular}{l l l l l l l }")
    print("\\toprule")

    header_row = "algorithm & " + " & ".join(dataset_short_names) + " \\\\"
    print(header_row)
    print("\\midrule")

    for row in pivot_table.rows(named=True):
        algorithm_name = row["algorithm"]
        row_values = [algorithm_name]

        for ds in datasets:
            metric_value = row.get(ds)
            if metric_value is not None:
                row_values.append(str(int(round(metric_value))))
            else:
                row_values.append("-")

        row_str = " & ".join(row_values) + " \\\\"
        print(row_str)

    print("\\bottomrule")
    print("\\end{NiceTabular}")
    print("\\end{center}")
    print("\\end{table}")


if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--output", help="the path to the output directory", default="plots")
    aparser.add_argument(
        "--plot-type",
        help="type of plot (pareto, radar, difficulty, performance-gap, split-difficulties, critdiff, build-time-table, dataset-geometry-grid)",
        default="pareto",
    )
    aparser.add_argument("--dataset", help="dataset", default="agnews-mxbai-1024-euclidean")
    aparser.add_argument("--selected", help="plot results for only selected algorithms", action="store_true")
    aparser.add_argument("--gpu", help="plot results for GPU algorithms", action="store_true")
    aparser.add_argument("--pca", help="add PCA plot (only applicable for pareto plot)", action="store_true")
    aparser.add_argument("--recall", help="recall level for plot", default=0.95)
    aparser.add_argument("--count", help="number of nearest neighbors (k) to use", default=100)

    args = aparser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data_dir = pathlib.Path(args.results)
    out_dir = pathlib.Path(args.output)

    normalize_names = pl.col("dataset")

    pca_mahalanobis = pl.read_parquet(data_dir / "data-pca-mahalanobis.parquet").with_columns(normalize_names)

    if args.plot_type == "dataset-geometry-grid":
        dataset_geometry_grid(out_dir, pca_mahalanobis)
        sys.exit(0)

    query_stats = pl.read_parquet(data_dir / "stats.parquet").with_columns(normalize_names)

    if args.plot_type == "difficulty":
        plot_difficulty_ridgeline(out_dir, query_stats)
        sys.exit(0)

    if args.plot_type == "filtered-difficulty":
        plot_filtered_rc_ridgeline(out_dir, query_stats, data_dir=pathlib.Path("data"))
        sys.exit(0)

    summary = pl.read_parquet(data_dir / "summary.parquet").with_columns(normalize_names)
    detail = pl.concat([pl.read_parquet(path) for path in data_dir.glob("*__detail.parquet")]).with_columns(
        normalize_names
    )

    datasets = args.dataset.split(",")
    count = int(args.count)
    recall = float(args.recall)

    if datasets[0].endswith("-binary"):
        point_type = "binary"
        distance_metric = "hamming"
    elif datasets[0].endswith("-uint8"):
        point_type = "uint8"
        distance_metric = "euclidean"
    else:
        point_type = "float"
        distance_metric = "normalized"

    definitions = get_definitions(
        dimension=None,
        point_type=point_type,
        distance_metric=distance_metric,
        count=count,
        base_dir="vibe/algorithms",
    )

    definitions = filter_disabled_algorithms(definitions)
    definitions = filter_algorithms_by_device(definitions, args.gpu)

    all_algorithms = list(sorted(set(definition.algorithm for definition in definitions)))

    if args.selected:
        algorithms = [
            "glass",
            "hnswlib",
            "ivfpqfs(faiss)",
            "lorann",
            "ngt-onng",
            "ngt-qg",
            "scann",
            "symphonyqg",
            "vamana-lvq(svs)",
        ]
    else:
        algorithms = all_algorithms

    if args.plot_type == "pareto":
        if args.gpu:
            ylim = (2e3, 3e5)
        else:
            ylim = (1e1, 1.1e4)

        if "llama-128-ip" in datasets or "yi-128-ip" in datasets:
            xlim = (0, 1)
        else:
            xlim = (0.7, 1.0)

        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=pca_mahalanobis if args.pca else None,
            datasets=datasets,
            algorithms=algorithms,
            k=count,
            xlim=xlim,
            ylim=ylim,
            separate_legend=False,
            gpu=args.gpu,
        )
    elif args.plot_type == "radar":
        radar_at_recall_plot(
            out_dir,
            summary,
            query_stats,
            algorithms=algorithms,
            all_algorithms=all_algorithms,
            height=len(algorithms) / 2,
            recall=recall,
            k=count,
            gpu=args.gpu,
        )
    elif args.plot_type == "performance-gap":
        if len(datasets) != 2:
            raise ValueError("plot type performance-gap requires two datasets")

        performance_gap_plot(
            out_dir,
            datasets[0],
            datasets[1],
            summary,
            pca_mahalanobis,
            k=count,
            recall=recall,
            gpu=args.gpu,
        )
    elif args.plot_type == "split-difficulties":
        split_difficulties_plot(
            out_dir,
            summary,
            detail,
            query_stats,
            recall,
            datasets=datasets,
            k=count,
            gpu=args.gpu,
        )
    elif args.plot_type == "critdiff":
        latency_difference_plot(
            summary,
            detail,
            float(args.recall),
            ID_DATASETS + OOD_DATASETS,
            output=args.output,
            algorithms=algorithms,
            k=count,
            gpu=args.gpu,
        )
    elif args.plot_type == "build-time-table":
        print_metric_table(summary, recall=recall, k=count, algorithms=algorithms, metric="build_time")
    elif args.plot_type == "index-size-table":
        print_metric_table(summary, recall=recall, k=count, algorithms=algorithms, metric="index_size")
    elif args.plot_type == "paper":
        paper(out_dir, all_algorithms, summary, detail, query_stats, pca_mahalanobis)
    else:
        raise ValueError(f"invalid plot type: {args.plot_type}")
