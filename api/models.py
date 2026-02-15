"""Pydantic request/response schemas for the text-to-sql API."""

from pydantic import BaseModel, Field


# --- Request schemas ---


class GenerateRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the Chinook database")
    model: str = Field(default="qwen2.5-coder-32b", description="Model key to use for generation")
    execute: bool = Field(default=True, description="Whether to execute the generated SQL")
    ground_truth_sql: str | None = Field(default=None, description="Optional ground truth SQL for accuracy comparison")


class EvaluateRequest(BaseModel):
    models: list[str] = Field(..., description="List of model keys to evaluate")
    query_ids: list[str] | None = Field(default=None, description="Optional filter: only evaluate these query IDs")
    concept_tags: list[str] | None = Field(default=None, description="Optional filter: only evaluate these concept tags")


# --- Response schemas ---


class ExecutionResult(BaseModel):
    success: bool
    columns: list[str] | None = None
    rows: list[list] | None = None
    error: str | None = None


class ComparisonResult(BaseModel):
    exact_match: bool
    fuzzy_match: bool
    row_count_match: bool
    row_overlap_pct: float
    gt_count: int
    gen_count: int


class GenerateResponse(BaseModel):
    model: str
    question: str
    generated_sql: str | None
    execution: ExecutionResult | None = None
    comparison: ComparisonResult | None = None
    latency_s: float | None = None
    error: str | None = None


class ModelSummary(BaseModel):
    total: int
    execution_rate: float
    exact_match_rate: float
    accuracy: float
    avg_row_overlap_pct: float
    failure_modes: dict[str, int]


class EvaluateResponse(BaseModel):
    summary: dict[str, ModelSummary]
    wandb_run_url: str | None = None


class ModelInfo(BaseModel):
    key: str
    huggingface_id: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    available_models: list[ModelInfo]
    wandb_enabled: bool
