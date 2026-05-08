#!/usr/bin/env python3
"""
Interactive VIBE filtered dataset explorer.

Shows PC1/PC2 scatter with the individual query and filtered subset highlighted,
and lets you sweep across selectivity levels.

Usage:
    cd /pub/scratch/twuebker/vibe
    source .venv/bin/activate
    python explore.py [--port 8050]
Then open http://localhost:8050 in a browser (port-forward if on a remote server).
"""

import argparse
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
WIKI_SRC = Path("/pub/scratch/twuebker/data/wiki_15.4M")
PCA_SAMPLE_SIZE = 2000
RNG_SEED = 1234


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_pca_sample_indices(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the exact random sample used by export_results.py to build the PCA."""
    gen = np.random.default_rng(RNG_SEED)
    mahal_n = min(n_train, 100_000)
    gen.choice(n_train, mahal_n, replace=False)          # consume mahalanobis draw
    train_idx = np.sort(gen.choice(n_train, min(n_train, PCA_SAMPLE_SIZE), replace=False))
    test_idx = np.sort(gen.choice(n_test, min(n_test, PCA_SAMPLE_SIZE), replace=False))
    return train_idx, test_idx


def pca_xy(pca_df: pl.DataFrame, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y) arrays for a dataset."""
    d = pca_df.filter(pl.col("dataset") == dataset)
    tr = d.filter(pl.col("part") == "train")
    te = d.filter(pl.col("part") == "test")
    return tr["x"].to_numpy(), tr["y"].to_numpy(), te["x"].to_numpy(), te["y"].to_numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class AppData:
    """Loads all data required by the app at startup."""

    def __init__(self):
        t0 = time.time()
        print("Loading PCA data …")
        self.pca_df = pl.read_parquet(RESULTS_DIR / "data-pca-mahalanobis.parquet")
        print(f"  done ({time.time()-t0:.1f}s)")

        self.workloads: dict = {}
        self._load_wiki()
        self._load_arxiv()
        self._load_yfcc()
        print(f"All data ready ({time.time()-t0:.1f}s total)")

    # ------------------------------------------------------------------
    # Wiki
    # ------------------------------------------------------------------
    def _load_wiki(self):
        t = time.time()
        print("Loading wiki data …")
        dataset = "wiki_1M"

        base = pl.read_parquet(DATA_DIR / "wiki_1M_base_metadata.parquet")
        chunk_ids = base["chunk_id"].to_numpy()
        wiki_ids  = base["wiki_id"].to_numpy()
        n_train   = len(base)

        query_meta = pl.read_parquet(DATA_DIR / "wiki_1M_query_metadata.parquet")

        with h5py.File(DATA_DIR / "wiki_1M.hdf5") as f:
            n_test = f["test"].shape[0]  # 100

        train_idx, _ = reconstruct_pca_sample_indices(n_train, n_test)
        tx, ty, qx, qy = pca_xy(self.pca_df, dataset)

        # ------ persons table for correlated workloads ------
        print("  loading persons …")
        import pandas as pd
        persons = (
            pl.from_pandas(pd.read_csv(WIKI_SRC / "persons.csv"))
            .with_columns(
                pl.col("wiki_id").cast(pl.Int32),
                pl.col("birth_date").str.to_date(format="%Y-%m-%d", strict=False),
            )
            .drop_nulls("birth_date")
        )
        all_person_wiki_ids = persons["wiki_id"].to_numpy()
        mask_all_persons = np.isin(wiki_ids, all_person_wiki_ids)

        # ------ Uncorrelated ------
        unc_meta = (
            query_meta
            .filter(pl.col("correlation_type") == "uncorrelated")
            .sort("local_query_id")
        )
        fc_list = json.loads(unc_meta["filter_conditions"][0])
        unc_sels = {}
        for cond in fc_list:
            m = re.search(r"chunk_id\s*<\s*(\d+)", cond)
            if not m:
                # 100 % case (no WHERE clause)
                threshold = n_train + 1
                mask_full = np.ones(n_train, dtype=bool)
            else:
                threshold = int(m.group(1))
                mask_full = chunk_ids < threshold
            actual_sel = mask_full.mean() * 100
            key = f"{min(actual_sel, 100):.1f}%"
            unc_sels[key] = {
                "mask_pca": mask_full[train_idx],
                "sel_pct":  min(actual_sel, 100.0),
            }
        # add explicit 100% if missing
        if "100.0%" not in unc_sels:
            unc_sels["100.0%"] = {
                "mask_pca": np.ones(len(train_idx), dtype=bool),
                "sel_pct":  100.0,
            }
        unc_sel_keys = sorted(unc_sels, key=lambda k: unc_sels[k]["sel_pct"])
        # keep only up to 100%
        unc_sel_keys = [k for k in unc_sel_keys if unc_sels[k]["sel_pct"] <= 100.0]

        self.workloads["wiki_uncorrelated"] = {
            "label":       "Wiki – Uncorrelated",
            "dataset":     dataset,
            "n_queries":   50,
            "pca_train_x": tx,
            "pca_train_y": ty,
            # global_query_id 0-49  →  test rows 0-49
            "query_pca_x": qx[:50],
            "query_pca_y": qy[:50],
            "sel_keys":    unc_sel_keys,
            "sel_data":    unc_sels,
            "per_query_filter": False,
        }

        # ------ Helper: date-based filter mask ------
        def date_mask(start_str, end_str):
            filtered = persons.filter(
                (pl.col("birth_date") >= pl.lit(start_str).str.to_date())
                & (pl.col("birth_date") < pl.lit(end_str).str.to_date())
            )["wiki_id"].to_numpy()
            return np.isin(wiki_ids, filtered)

        # parse date ranges from first neg_correlated query
        neg_meta = (
            query_meta
            .filter(pl.col("correlation_type") == "neg_correlated")
            .sort("local_query_id")
        )
        fc_neg = json.loads(neg_meta["filter_conditions"][0])

        # parse date ranges from first pos_correlated query (same date ranges)
        pos_meta = (
            query_meta
            .filter(pl.col("correlation_type") == "pos_correlated")
            .sort("local_query_id")
        )
        fc_pos = json.loads(pos_meta["filter_conditions"][0])

        def build_date_sels(fc_strs):
            sels = {}
            pat = r"birth_date >= date\('([^']+)'\) AND p\.birth_date < date\('([^']+)'\)"
            for cond in fc_strs:
                m = re.search(pat, cond)
                if m:
                    start, end = m.group(1), m.group(2)
                    mask = date_mask(start, end)
                else:
                    # no date restriction → all persons
                    mask = mask_all_persons
                actual_sel = mask.mean() * 100
                key = f"{actual_sel:.1f}%"
                sels[key] = {
                    "mask_pca": mask[train_idx],
                    "sel_pct":  actual_sel,
                }
            # add "all persons" entry
            ap_sel = mask_all_persons.mean() * 100
            key = f"{ap_sel:.1f}%"
            if key not in sels:
                sels[key] = {
                    "mask_pca": mask_all_persons[train_idx],
                    "sel_pct":  ap_sel,
                }
            return sels

        neg_sels = build_date_sels(fc_neg)
        pos_sels = build_date_sels(fc_pos)
        neg_sel_keys = sorted(neg_sels, key=lambda k: neg_sels[k]["sel_pct"])
        pos_sel_keys = sorted(pos_sels, key=lambda k: pos_sels[k]["sel_pct"])

        # neg_correlated: global_query_id 50-99  →  test rows 50-99
        self.workloads["wiki_neg_correlated"] = {
            "label":       "Wiki – Neg. Correlated",
            "dataset":     dataset,
            "n_queries":   50,
            "pca_train_x": tx,
            "pca_train_y": ty,
            "query_pca_x": qx[50:100],
            "query_pca_y": qy[50:100],
            "sel_keys":    neg_sel_keys,
            "sel_data":    neg_sels,
            "per_query_filter": False,
        }

        # pos_correlated: global_query_id == -1  →  no PCA positions
        self.workloads["wiki_pos_correlated"] = {
            "label":       "Wiki – Pos. Correlated",
            "dataset":     dataset,
            "n_queries":   50,
            "pca_train_x": tx,
            "pca_train_y": ty,
            "query_pca_x": None,   # query vectors not in HDF5 test set
            "query_pca_y": None,
            "sel_keys":    pos_sel_keys,
            "sel_data":    pos_sels,
            "per_query_filter": False,
        }
        print(f"  wiki done ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # arXiv
    # ------------------------------------------------------------------
    def _load_arxiv(self):
        t = time.time()
        print("Loading arXiv data …")
        dataset = "arxiv_1M"

        base = pl.read_parquet(DATA_DIR / "arxiv_1M_base_metadata.parquet")
        arxiv_ids = base["arxiv_id"].to_numpy()
        n_train   = len(base)

        query_meta = pl.read_parquet(DATA_DIR / "arxiv_1M_query_metadata.parquet")

        with h5py.File(DATA_DIR / "arxiv_1M.hdf5") as f:
            n_test = f["test"].shape[0]  # 50

        train_idx, _ = reconstruct_pca_sample_indices(n_train, n_test)
        tx, ty, qx, qy = pca_xy(self.pca_df, dataset)

        # One row per (query_id, selectivity_pct) → pick unique selectivities
        arxiv_sels = {}
        for row in (
            query_meta
            .unique(["selectivity_pct", "id_threshold"])
            .sort("selectivity_pct")
            .iter_rows(named=True)
        ):
            sel_pct   = float(row["selectivity_pct"])
            threshold = int(row["id_threshold"])
            mask = (arxiv_ids < threshold) if threshold >= 0 else np.ones(n_train, dtype=bool)
            key = f"{sel_pct:.0f}%"
            arxiv_sels[key] = {
                "mask_pca": mask[train_idx],
                "sel_pct":  mask.mean() * 100,
            }
        sel_keys = sorted(arxiv_sels, key=lambda k: arxiv_sels[k]["sel_pct"])

        self.workloads["arxiv"] = {
            "label":       "arXiv",
            "dataset":     dataset,
            "n_queries":   n_test,
            "pca_train_x": tx,
            "pca_train_y": ty,
            "query_pca_x": qx[:n_test],
            "query_pca_y": qy[:n_test],
            "sel_keys":    sel_keys,
            "sel_data":    arxiv_sels,
            "per_query_filter": False,
        }
        print(f"  arXiv done ({time.time()-t:.1f}s)")

    # ------------------------------------------------------------------
    # YFCC  (per-query tag-intersection filter)
    # ------------------------------------------------------------------
    def _load_yfcc(self):
        t = time.time()
        print("Loading YFCC data …")
        dataset = "yfcc_1M"

        base = pl.read_parquet(DATA_DIR / "yfcc_1M_base_metadata.parquet")
        n_train = len(base)

        query_meta = pl.read_parquet(DATA_DIR / "yfcc_1M_query_metadata.parquet")

        with h5py.File(DATA_DIR / "yfcc_1M.hdf5") as f:
            n_test = f["test"].shape[0]  # 100 000

        train_idx, test_idx = reconstruct_pca_sample_indices(n_train, n_test)
        tx, ty, qx, qy = pca_xy(self.pca_df, dataset)

        # Build tag → vector_ids inverted index
        print("  building YFCC tag inverted index …")
        tag_to_vecs: dict[int, np.ndarray] = {}
        tmp: dict[int, list] = {}
        for vid, tags in enumerate(base["tag_ids"].to_list()):
            for tag in tags:
                if tag not in tmp:
                    tmp[tag] = []
                tmp[tag].append(vid)
        tag_to_vecs = {t: np.array(v, dtype=np.int32) for t, v in tmp.items()}
        del tmp

        # Cap to 50 queries: take the first 50 from the PCA-sampled set
        pca_query_ids = test_idx[:50].tolist()
        qid_to_pca_row = {qid: i for i, qid in enumerate(pca_query_ids)}

        # Precompute filter for the capped query set
        print("  precomputing YFCC filter masks for 50 sampled queries …")
        pca_qmeta = (
            query_meta
            .filter(pl.col("query_id").is_in(pca_query_ids))
            .sort("query_id")
        )
        query_filters: dict[int, dict] = {}
        for row in pca_qmeta.iter_rows(named=True):
            qid  = row["query_id"]
            tags = row["tag_ids"] or []
            if tags:
                parts = [tag_to_vecs[t] for t in tags if t in tag_to_vecs]
                if parts:
                    matching = np.unique(np.concatenate(parts))
                    mask = np.zeros(n_train, dtype=bool)
                    mask[matching] = True
                else:
                    mask = np.zeros(n_train, dtype=bool)
            else:
                mask = np.zeros(n_train, dtype=bool)
            query_filters[qid] = {
                "mask_pca": mask[train_idx],
                "sel_pct":  mask.mean() * 100,
                "tags":     tags,
            }

        self.workloads["yfcc"] = {
            "label":            "YFCC",
            "dataset":          dataset,
            "n_queries":        len(pca_query_ids),
            "pca_train_x":      tx,
            "pca_train_y":      ty,
            "query_pca_x":      qx[:len(pca_query_ids)],
            "query_pca_y":      qy[:len(pca_query_ids)],
            "pca_query_ids":    pca_query_ids,
            "query_filters":    query_filters,
            "per_query_filter": True,
            # no fixed selectivity levels
            "sel_keys":  [],
            "sel_data":  {},
        }
        print(f"  YFCC done ({time.time()-t:.1f}s)")


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

def build_app(data: AppData) -> dash.Dash:
    app = dash.Dash(__name__, title="VIBE Explorer")

    workload_options = [
        {"label": wl["label"], "value": key}
        for key, wl in data.workloads.items()
    ]

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "0 16px"},
        children=[
            html.H2("VIBE Filtered Dataset Explorer", style={"textAlign": "center", "margin": "12px 0"}),

            # ---- control row ----
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "flex-start", "flexWrap": "wrap"},
                children=[
                    html.Div([
                        html.Label("Workload", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="workload-dd",
                            options=workload_options,
                            value="wiki_uncorrelated",
                            clearable=False,
                            style={"minWidth": "240px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Query index", style={"fontWeight": "bold"}),
                        dcc.Slider(
                            id="query-slider",
                            min=0, max=49, step=1, value=0,
                            marks={i: str(i) for i in range(0, 50, 10)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                    ], style={"width": "320px"}),
                    html.Div([
                        html.Label("Selectivity", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="sel-dd",
                            clearable=False,
                            style={"minWidth": "120px"},
                        ),
                    ]),
                    html.Div(id="info-panel", style={"fontSize": "0.9em", "lineHeight": "1.6", "padding": "4px 0"}),
                ],
            ),

            # ---- main plot ----
            dcc.Graph(id="scatter", style={"height": "620px"}, config={"scrollZoom": True}),
        ],
    )

    # ------------------------------------------------------------------ #
    # Callback 1: update controls when workload changes                   #
    # ------------------------------------------------------------------ #
    @app.callback(
        [
            Output("sel-dd", "options"),
            Output("sel-dd", "value"),
            Output("query-slider", "max"),
            Output("query-slider", "marks"),
            Output("query-slider", "value"),
        ],
        Input("workload-dd", "value"),
    )
    def update_controls(workload_key):
        wl = data.workloads[workload_key]
        sel_keys = wl["sel_keys"]

        if sel_keys:
            sel_options = [{"label": k, "value": k} for k in sel_keys]
            default_sel = sel_keys[len(sel_keys) // 2]
        else:
            sel_options = [{"label": "per-query", "value": "per-query"}]
            default_sel = "per-query"

        n_q   = wl["n_queries"] - 1
        marks = {i: str(i) for i in range(0, n_q + 1, max(1, n_q // 8))}
        return sel_options, default_sel, n_q, marks, 0

    # ------------------------------------------------------------------ #
    # Callback 2: update scatter when any control changes                 #
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("scatter", "figure"), Output("info-panel", "children")],
        [
            Input("workload-dd", "value"),
            Input("query-slider", "value"),
            Input("sel-dd", "value"),
        ],
    )
    def update_scatter(workload_key, query_idx, sel_key):
        wl = data.workloads[workload_key]
        tx = wl["pca_train_x"]
        ty = wl["pca_train_y"]

        # ---------- determine filter mask ----------
        if wl["per_query_filter"]:
            # YFCC: per-query filter
            pca_qids = wl["pca_query_ids"]
            if query_idx < len(pca_qids):
                qid   = pca_qids[query_idx]
                fdata = wl["query_filters"].get(qid, {})
            else:
                fdata = {}
            mask_pca = fdata.get("mask_pca", np.zeros(len(tx), dtype=bool))
            sel_pct  = fdata.get("sel_pct", 0.0)
            tags     = fdata.get("tags", [])
            extra_info = f"Tags: {tags}"
        else:
            if sel_key and sel_key in wl["sel_data"]:
                sdata    = wl["sel_data"][sel_key]
                mask_pca = sdata["mask_pca"]
                sel_pct  = sdata["sel_pct"]
            else:
                mask_pca = np.ones(len(tx), dtype=bool)
                sel_pct  = 100.0
            extra_info = ""

        n_filtered = int(mask_pca.sum())
        n_total    = len(tx)

        # ---------- build traces ----------
        traces = []
        not_mask = ~mask_pca

        # unfiltered background (gray)
        if not_mask.any():
            traces.append(
                go.Scattergl(
                    x=tx[not_mask], y=ty[not_mask],
                    mode="markers",
                    marker=dict(size=2, color="rgba(160,160,160,0.35)"),
                    name="unfiltered",
                    hovertemplate="unfiltered<extra></extra>",
                )
            )

        # filtered subset (blue)
        if mask_pca.any():
            traces.append(
                go.Scattergl(
                    x=tx[mask_pca], y=ty[mask_pca],
                    mode="markers",
                    marker=dict(size=3, color="rgba(31,119,180,0.7)"),
                    name=f"filtered ({sel_pct:.1f}%)",
                    hovertemplate="filtered<extra></extra>",
                )
            )

        # query vector (red star)
        qpx = wl.get("query_pca_x")
        qpy = wl.get("query_pca_y")
        if qpx is not None and query_idx < len(qpx):
            traces.append(
                go.Scatter(
                    x=[float(qpx[query_idx])],
                    y=[float(qpy[query_idx])],
                    mode="markers",
                    marker=dict(size=16, color="crimson", symbol="star",
                                line=dict(color="darkred", width=1.5)),
                    name=f"query {query_idx}",
                    hovertemplate=f"Query {query_idx}<extra></extra>",
                )
            )
        elif wl["query_pca_x"] is None:
            # pos_correlated: no PCA position available
            traces.append(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=12, color="crimson", symbol="star"),
                    name="query (no PCA pos.)",
                )
            )

        fig = go.Figure(traces)
        fig.update_layout(
            title=dict(
                text=f"{wl['label']}  ·  Query {query_idx}  ·  Selectivity ≈ {sel_pct:.1f}%",
                font=dict(size=14),
            ),
            xaxis_title="PC1",
            yaxis_title="PC2",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(l=50, r=20, t=50, b=40),
            uirevision=workload_key,   # preserve zoom when only query/sel changes
        )

        info = html.Div([
            html.Div(f"Filtered pts (PCA sample): {n_filtered} / {n_total}"),
            html.Div(f"Actual selectivity: {sel_pct:.2f}%"),
            html.Div(extra_info, style={"color": "#555", "fontSize": "0.85em"}) if extra_info else html.Div(),
        ])
        if wl["query_pca_x"] is None:
            info = html.Div([info, html.Div("⚠ Query vectors not in HDF5 test set (pos. corr.)",
                                             style={"color": "orange"})])

        return fig, info

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print("Loading data … (first run may take ~30–60 s)")
    appdata = AppData()

    app = build_app(appdata)
    print(f"\nServing on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
