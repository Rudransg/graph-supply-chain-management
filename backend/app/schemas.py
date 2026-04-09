from pydantic import BaseModel, Field
from typing import Optional


# ── existing schemas (unchanged) ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    product_name: str = Field(
        ...,
        description="Product id such as 'AT5X5K' or a signal-product key such as 'production_unit_AT5X5K'"
    )

class PredictResponse(BaseModel):
    product_name: str
    product_idx: int
    prediction: dict[str, float] | float
    next_day_prediction: dict[str, float] | None = None
    daily_forecast: dict[str, list[float]] | None = None
    forecast_horizon_days: int | None = None
    model_version: str
    run_id: str
    lower_bound : Optional[dict] = None
    upper_bound : Optional[dict] = None

class MetricsResponse(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
    test_asymmetric_loss: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_nodes: int
    node_type: str
    conv_type: str
    layers: int

class ProductsResponse(BaseModel):
    products: dict[str, str]

class ScatterPoint(BaseModel):
    predicted: float
    actual: float

class AllPredsResponse(BaseModel):
    points: list[ScatterPoint]
    total: int


# ── NEW schemas ───────────────────────────────────────────────────────────────

class ProductDetail(BaseModel):
    product: str
    group: str
    subgroup: str
    factory: str
    forecast_kg: float
    trend_pct: float
    status: str

class ProductsListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    data: list[ProductDetail]

class FilterOptions(BaseModel):
    factories: list[str]
    groups: list[str]
    subgroups: list[str]

class FactoryLoad(BaseModel):
    factory: str
    total_forecast_kg: float
    product_count: int
    load_pct: float

class TrendResponse(BaseModel):
    product: str
    historical: list[float]
    forecast: list[float]
    days_historical: int
    days_forecast: int

class RelatedProduct(BaseModel):
    product: str
    forecast_kg: float
    trend_pct: float
    factory: str

class DashboardStats(BaseModel):
    total_products: int
    at_risk: int
    on_track: int
    active_factories: int
    forecast_accuracy: float
    new_at_risk_today: int

class GraphNode(BaseModel):
    id: str
    label: str
    factory: str
    group: str
    forecast_kg: float
    status: str

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

class WhatIfRequest(BaseModel):
    product_name:        str
    zeroed_products:     list[str]         = []
    zeroed_factories:    list[str]         = []
    capacity_overrides:  dict[str, float]  = {}
    dropped_relations:   list[str]         = []

class WhatIfResponse(BaseModel):
    product_name: str
    baseline:     dict
    scenario:     dict
    delta:        dict   # absolute difference
    delta_pct:    dict   # % difference