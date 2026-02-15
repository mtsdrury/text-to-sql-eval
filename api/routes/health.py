"""GET /health — availability check."""

from fastapi import APIRouter

from src.llm_generator import MODELS
from src.query_runner import execute_query
from api.models import HealthResponse, ModelInfo
from api import wandb_logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    db_check = execute_query("SELECT 1")
    return HealthResponse(
        status="ok",
        db_connected=db_check["success"],
        available_models=[
            ModelInfo(key=k, huggingface_id=v) for k, v in MODELS.items()
        ],
        wandb_enabled=wandb_logger.is_enabled(),
    )
