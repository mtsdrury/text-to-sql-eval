"""FastAPI application — wires routers and manages W&B lifecycle."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from api import wandb_logger
from api.models import ModelInfo
from api.routes import health, generate, evaluate
from src.llm_generator import MODELS

# Load .env from project root (works both locally and in Docker)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    enabled = wandb_logger.init_run()
    print(f"W&B logging: {'enabled' if enabled else 'disabled (no WANDB_API_KEY)'}")
    yield
    # Shutdown
    wandb_logger.finish()


app = FastAPI(
    title="Text-to-SQL Evaluation API",
    description="Generate and evaluate SQL from natural language using open-source LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(evaluate.router)


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    return [ModelInfo(key=k, huggingface_id=v) for k, v in MODELS.items()]
