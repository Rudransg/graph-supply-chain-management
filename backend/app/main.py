from linecache import cache
import sys
import os
from pathlib import Path
import string

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .forecast_data import (
    get_trend_for_product,
    list_forecast_products,
    get_products_by_category,
    get_category_forecast_summary,
)
from .schemas import (
    PredictRequest, PredictResponse,
    MetricsResponse, HealthResponse,
    ProductsResponse, AllPredsResponse, ScatterPoint,
    ProductsListResponse, ProductDetail, FilterOptions,
    FactoryLoad, TrendResponse, RelatedProduct,
    DashboardStats, GraphNode, GraphEdge, GraphResponse,
     WhatIfRequest, WhatIfResponse,   # ← add these two
)
from .model_runtime import (
    model_service,
    NODE_TYPE,
    CONV_TYPE,
    LAYERS,
    NUM_NODES,
    TARGET_SIGNALS,
    TEMPORAL_FEATURE_DIM,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    AGGREGATION,
    ROLLING_WINDOW,
    EPOCHS,
    LEARNING_RATE,
    LOSS_ALPHA,
)
from .data_service import (  # was .data_service
    get_factory_map_multi, get_group_subgroup_map,
    build_forecast_cache, build_trend_series, get_edge_data
)
from src.logger import get_logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = get_logger(__name__)
# ── AFTER IMPORTS, BEFORE app = FastAPI() ─────
def get_plant_label(i: int) -> str:   # ← ADD THIS FUNCTION HERE
    letters = string.ascii_uppercase
    if i < 26:
        return f"Plant {letters[i]}"
    return f"Plant {letters[i // 26 - 1]}{letters[i % 26]}"

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Supply Graph — HeteroSAGE Inference API",
    version="2.0.0",
    description="FastAPI backend for SupplyGraphModel demand prediction with dashboard UI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://graph-supply-chain-management-5ggk.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── cache ─────────────────────────────────────────────────────────────────────

_forecast_cache: list[dict] = []


@app.on_event("startup")
def _warm_up():
    global _forecast_cache
    _forecast_cache = build_forecast_cache(
        idx_to_product=model_service.idx_to_product,
        predictions=model_service.predictions,
        targets=model_service.targets,
        values=model_service.values,
    )
    logger.info(f"[Startup] Forecast cache built for {len(_forecast_cache)} products")


def _cache() -> list[dict]:
    if not _forecast_cache:
        return build_forecast_cache(
            idx_to_product=model_service.idx_to_product,
            predictions=model_service.predictions,
            targets=model_service.targets,
            values=model_service.values,
        )
    return _forecast_cache

# ═════════ EXISTING ROUTES (unchanged) ═════════

@app.get("/")
def root():
    return {
        "message": "Supply Graph Inference API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "device": str(model_service.device),
        "num_nodes": NUM_NODES,
        "node_type": NODE_TYPE,
        "conv_type": CONV_TYPE,
        "layers": LAYERS,
    }


@app.get("/products", response_model=ProductsResponse)
def get_products():
    try:
        return {"products": model_service.get_all_products()}
    except Exception as e:
        logger.error(f"/products failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = model_service.predict(req.product_name)
        return {
            **result,
            "model_version": "hetero_sage_v2",
            "run_id": "a4a2ee5b17584655a4a01b60eea5a9d9",
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"/predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        m = model_service.metrics
        if not m:
            raise HTTPException(status_code=404, detail="metrics.json not found")
        return {
            "mae": float(m.get("mae", 0.0)),
            "mse": float(m.get("mse", 0.0)),
            "rmse": float(m.get("rmse", 0.0)),
            "r2": float(m.get("r2", 0.0)),
            "test_asymmetric_loss": float(m.get("test_asymmetric_loss", 0.0)),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/all", response_model=AllPredsResponse)
def all_predictions():
    try:
        points = model_service.get_scatter_data(limit=300)
        if not points:
            raise HTTPException(
                status_code=404,
                detail="predictions.npy / targets.npy not found"
            )
        return {"points": [ScatterPoint(**p) for p in points], "total": len(points)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/predictions/all failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    return {
        "node_type": NODE_TYPE,
        "num_nodes": NUM_NODES,
        "conv_type": CONV_TYPE,
        "layers": LAYERS,
        "in_channels": TEMPORAL_FEATURE_DIM,
        "hidden_channels": HIDDEN_CHANNELS,
        "out_channels": OUT_CHANNELS,
        "aggregation": AGGREGATION,
        "edge_relations": [
            "same_plant",
            "same_storage",
            "same_product_group",
            "same_product_subgroup",
        ],
        "target_signals": TARGET_SIGNALS,
        "rolling_window": ROLLING_WINDOW,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "loss_alpha": LOSS_ALPHA,
    }

# ═════════ NEW ROUTES (dashboard API) ═════════

@app.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats():
    cache = _cache()
    at_risk = [p for p in cache if p["status"] == "at_risk"]
    factories = factories = set(f for p in cache for f in p.get("factories", []))
    m = model_service.metrics
    r2 = float(m.get("r2", 0.9923)) if m else 0.9923

    return {
        "total_products": len(cache),
        "at_risk": len(at_risk),
        "on_track": len(cache) - len(at_risk),
        "active_factories": len(factories),
        "forecast_accuracy": round(r2 * 100, 1),
        "new_at_risk_today": max(0, len(at_risk) - 4),
    }


@app.get("/products/list", response_model=ProductsListResponse)
def list_products(
    search: str = Query(""),
    factory: str = Query(""),
    group: str = Query(""),
    status: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    cache = _cache()
    rows = cache.copy()

    if search:
        rows = [r for r in rows if search.lower() in r["product"].lower()]
    if factory:
        rows = [r for r in rows if factory in r.get("factories", [])]
    if group:
        rows = [r for r in rows if r["group"] == group]
    if status:
        rows = [r for r in rows if r["status"] == status]

    total = len(rows)
    start = (page - 1) * page_size
    end = start + page_size
    data = rows[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
        "data": [ProductDetail(**d) for d in data],
    }


@app.get("/products/filters", response_model=FilterOptions)
def get_filter_options():
    cache = _cache()
    factories = sorted(set(f for p in cache for f in p.get("factories", [])))
    groups = sorted(set(p["group"] for p in cache))
    subgroups = sorted(set(p["subgroup"] for p in cache))
    return {
        "factories": factories,
        "groups": groups,
        "subgroups": subgroups,
    }


@app.get("/products/at-risk")
def at_risk_products(top_n: int = Query(6)):
    cache = _cache()
    at_risk = [p for p in cache if p["status"] == "at_risk"]
    # Sort by worst trend across all signals
    def get_worst_trend(p):
        trends = [v for k, v in p.items() if k.endswith("_trend_pct")]
        return min(trends) if trends else 0
    at_risk.sort(key=get_worst_trend)
    return at_risk[:top_n]


@app.get("/products/{name}/related")
def related_products(name: str):
    cache = _cache()
    target = next((p for p in cache if p["product"] == name), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Product '{name}' not found")

    target_factories = set(target.get("factories", []))
    related = []
    for p in cache:
        if set(p.get("factories", [])) & target_factories and p["product"] != name:
            total_forecast = sum(v for k, v in p.items() if k.endswith("_forecast"))
            trends = [v for k, v in p.items() if k.endswith("_trend_pct")]
            worst_trend = min(trends) if trends else 0
            related.append(
                RelatedProduct(
                    product=p["product"],
                    forecast_kg=round(total_forecast, 2),
                    trend_pct=round(worst_trend, 1),
                    factory=", ".join(p.get("factories", [])),  # ← fixed
                )
            )
    return related[:5]

@app.post("/predict/whatif", response_model=WhatIfResponse)
def predict_whatif(req: WhatIfRequest):
    try:
        baseline = model_service.predict(req.product_name)
        scenario = model_service.predict_whatif(
            product_name=req.product_name,
            zeroed_products=req.zeroed_products,
            zeroed_factories=req.zeroed_factories,
            capacity_overrides=req.capacity_overrides,
            dropped_relations=req.dropped_relations,
        )

        baseline_signals = baseline["prediction"]
        scenario_signals = scenario["scenario"]

        delta = {
            signal: round(scenario_signals[signal] - baseline_signals[signal], 0)
            for signal in TARGET_SIGNALS
            if signal in scenario_signals and signal in baseline_signals
        }

        delta_pct = {
            signal: round(
                ((scenario_signals[signal] - baseline_signals[signal])
                 / max(baseline_signals[signal], 1)) * 100, 1
            )
            for signal in TARGET_SIGNALS
            if signal in scenario_signals and signal in baseline_signals
        }

        return {
            "product_name": req.product_name,
            "baseline": baseline_signals,
            "scenario": scenario_signals,
            "delta": delta,
            "delta_pct": delta_pct,
        }
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"/predict/whatif failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/factory/load")
def factory_load():
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    cache = _cache()
    # product → list of factories
    product_map = {p["product"]: p.get("factories", []) for p in cache}

    for product_id in model_service.product_ids:
        try:
            result        = model_service.predict(product_id)
            product_total = sum(result["prediction"].values())
            factories     = product_map.get(product_id, [])

            if not factories:
                continue

            # split load equally across all factories this product belongs to
            share = product_total / len(factories)
            for factory in factories:
                totals[factory] = totals.get(factory, 0.0) + share
                counts[factory] = counts.get(factory, 0) + 1

        except Exception as e:
            logger.warning(f"Skipping {product_id} in factory load: {e}")
            continue

    max_load = max(totals.values()) if totals else 0.0
    if max_load <= 0:
        return [FactoryLoad(factory=f, total_forecast_kg=round(totals[f], 2),
                            product_count=counts[f], load_pct=0.0)
                for f in sorted(totals)]

    return [FactoryLoad(factory=f, total_forecast_kg=round(totals[f], 2),
                        product_count=counts[f],
                        load_pct=round((totals[f] / max_load) * 100.0, 1))
            for f in sorted(totals)]

@app.get("/factories")
def get_factories():
    import pandas as pd
    from config.path_config import EDGES_PLANT_FILE

    df = pd.read_csv(EDGES_PLANT_FILE)
    unique_plants = sorted(df["Plant"].unique())
    factories = [get_plant_label(i) for i in range(len(unique_plants))]
    return {"factories": factories}

@app.get("/graph-edges")
def get_graph_edges():
    import json
    import pandas as pd
    from config.path_config import (
        PRODUCT_IDX_TO_NAME_PATH,
        EDGES_PLANT_FILE,
        EDGES_STORAGE_FILE,
        EDGES_GROUP_FILE,
        EDGES_SUBGROUP_FILE,
    )

    with open(PRODUCT_IDX_TO_NAME_PATH, "r") as f:
        idx_to_product = json.load(f)
    print("idx_to_product sample:", list(idx_to_product.items())[:5])
    df_plant = pd.read_csv(EDGES_PLANT_FILE)
    print("idx_to_product count:", len(idx_to_product))
    print("df_plant shape:", df_plant.shape)
    print("df_plant columns:", df_plant.columns.tolist())
    print("df_plant sample:", df_plant.head(2).to_dict())
    
    # idx_to_product has 3 entries per product (one per signal)
    # edge files use product-level indices → divide by NUM_SIGNALS
    product_idx_to_name = {int(k): v for k, v in idx_to_product.items()}

    print("product_idx_to_name keys:", sorted(product_idx_to_name.keys()))
    print("sample:", list(product_idx_to_name.items())[:5])
    df_plant = pd.read_csv(EDGES_PLANT_FILE)
    unique_plants = sorted(df_plant["Plant"].unique())
    plant_rename = {str(int(p)): get_plant_label(i) for i, p in enumerate(unique_plants)}

    def load_edges(filepath, edge_type):          # ← INDENTED inside function
        edges = []
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            src = product_idx_to_name.get(int(row["node1"]))
            tgt = product_idx_to_name.get(int(row["node2"]))
            if src and tgt:
                edge = {"source": src, "target": tgt, "type": edge_type}
                if edge_type == "same_storage":
                    edge["storage"] = str(row["Storage Location"])
                if edge_type == "same_plant":
                    edge["plant"] = plant_rename.get(str(int(row["Plant"])), str(int(row["Plant"])))
                edges.append(edge)
        return edges

    edges = []
    edges += load_edges(EDGES_PLANT_FILE,    "same_plant")
    edges += load_edges(EDGES_STORAGE_FILE,  "same_storage")
    edges += load_edges(EDGES_GROUP_FILE,    "same_product_group")
    edges += load_edges(EDGES_SUBGROUP_FILE, "same_product_subgroup")

    return {"edges": edges}

@app.get("/factory/graph", response_model=GraphResponse)
def factory_graph(
    same_plant: bool = Query(True),
    same_storage: bool = Query(True),
    same_group: bool = Query(False),
    same_subgroup: bool = Query(False),
):
    cache = _cache()
    edges_raw = get_edge_data()

    nodes_list = [
        GraphNode(
            id=p["product"],
            label=p["product"],
            factory=", ".join(p.get("factories", [])),  # join list to string for display
            group=p["group"],
            forecast_kg=sum(v for k, v in p.items() if k.endswith("_forecast")),
            status=p["status"],
        )
        for p in cache
    ]

    selected_relations: list[str] = []
    if same_plant:
        selected_relations.append("same_plant")
    if same_storage:
        selected_relations.append("same_storage")
    if same_group:
        selected_relations.append("same_product_group")
    if same_subgroup:
        selected_relations.append("same_product_subgroup")

    edges_list: list[GraphEdge] = []
    for rel in selected_relations:
        if rel not in edges_raw:
            continue
        df = edges_raw[rel]
        for _, row in df.iterrows():
            src = str(row.iloc[0])
            dst = str(row.iloc[1])
            edges_list.append(GraphEdge(source=src, target=dst, relation=rel))

    return {
        "nodes": nodes_list,
        "edges": edges_list,
    }

# ── forecast endpoints ────────────────────────────────────────────────────────

@app.get("/forecast/products")
def get_forecast_products():
    return {"products": list_forecast_products()}


@app.get("/forecast/trend/{product_id}")
def forecast_trend(
    product_id: str,
    signal_type: str = "production_unit",
    limit: int = 30,
):
    points = get_trend_for_product(product_id, signal_type=signal_type, limit=limit)
    if not points:
        raise HTTPException(
            status_code=404,
            detail="No data for this product and signal type",
        )
    return {
        "product": product_id,
        "signal_type": signal_type,
        "points": points,
    }


@app.get("/forecast/live/{product_id}")
def forecast_live_trend(
    product_id: str,
    signal_type: str = "production_unit",
    history_points: int = Query(30, ge=1, le=120),
):
    try:
        return model_service.get_live_forecast_series(
            product_name=product_id,
            signal_type=signal_type,
            history_points=history_points,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"/forecast/live failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/category/{category}")
def forecast_by_category(category: str):
    """
    Category can be: 'production', 'delivery', or 'supply_order'.
    """
    valid = {"production", "delivery", "supply_order"}
    if category not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"category must be one of {sorted(valid)}",
        )

    products = get_products_by_category(category)
    summary = get_category_forecast_summary(category)

    return {
        "category": category,
        "total": len(products),
        "products": [{"product": p} for p in products],
        "summary": summary,
    }
