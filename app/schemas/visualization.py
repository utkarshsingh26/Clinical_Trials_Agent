from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, Field


class VizType(str, Enum):
    BAR_CHART = "bar_chart"
    GROUPED_BAR_CHART = "grouped_bar_chart"
    TIME_SERIES = "time_series"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    NETWORK_GRAPH = "network_graph"
    PIE_CHART = "pie_chart"


# --- Encoding types (discriminated union on viz_type) ---

class AxisField(BaseModel):
    """Maps a visual channel to a data field."""
    field: str = Field(..., description="Key in the data array.")
    label: Optional[str] = Field(None, description="Human-readable axis label.")
    type: Literal["quantitative", "ordinal", "nominal", "temporal"] = Field(
        "nominal",
        description="Data type hint for the frontend renderer.",
    )


class CartesianEncoding(BaseModel):
    """
    Encoding for bar_chart, grouped_bar_chart, time_series, scatter, histogram.
    Maps x/y axes and an optional series grouping field.
    """
    encoding_type: Literal["cartesian"] = "cartesian"
    x: AxisField
    y: AxisField
    series: Optional[AxisField] = Field(
        None,
        description="Grouping field for grouped bar charts or multi-line time series.",
    )
    sort: Optional[Literal["ascending", "descending", "none"]] = Field(
        "descending",
        description="Sort order for ordinal/nominal x-axis values.",
    )


class NodeDef(BaseModel):
    """A node in a network graph."""
    id: str
    label: str
    type: str = Field(..., description="Entity type: drug, sponsor, condition, investigator, site.")
    weight: Optional[float] = Field(None, description="Optional size/importance weight.")
    properties: dict[str, Any] = Field(default_factory=dict)


class EdgeDef(BaseModel):
    """An edge in a network graph."""
    source: str = Field(..., description="Node id.")
    target: str = Field(..., description="Node id.")
    weight: Optional[float] = Field(None, description="Edge strength, e.g. number of co-occurring trials.")
    label: Optional[str] = None


class NetworkEncoding(BaseModel):
    """
    Encoding for network_graph visualizations.
    Nodes are entities (drugs, sponsors, conditions).
    Edges are relationships between them.
    """
    encoding_type: Literal["network"] = "network"
    nodes: list[NodeDef]
    edges: list[EdgeDef]
    node_color_by: Optional[str] = Field(
        None,
        description="Node field to use for color grouping, e.g. 'type'.",
    )
    edge_weight_label: Optional[str] = Field(
        None,
        description="Human-readable label for what edge weight represents.",
    )


VisualizationEncoding = Annotated[
    Union[CartesianEncoding, NetworkEncoding],
    Field(discriminator="encoding_type"),
]


# --- Top-level visualization spec ---

class VisualizationSpec(BaseModel):
    """
    The core output: a frontend-renderable visualization specification.
    Does not contain rendered output — a frontend consumes this to draw the chart.
    """
    type: VizType
    title: str = Field(..., description="Human-readable chart title.")
    encoding: VisualizationEncoding
    data: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Array of data point objects. Each object maps field names "
            "from encoding to values. For network graphs this is empty — "
            "nodes/edges live in the encoding."
        ),
    )
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    color_scheme: Optional[str] = Field(
        None,
        description="Optional color scheme hint for the renderer, e.g. 'categorical', 'sequential'.",
    )