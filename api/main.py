"""FastAPI application — wires routers and serves the SQL generation playground."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from api.models import ModelInfo
from api.routes import health, generate
from src.llm_generator import MODELS

# Load .env from project root (works both locally and in Docker)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


app = FastAPI(
    title="Text-to-SQL Evaluation API",
    description="Generate and evaluate SQL from natural language using open-source LLMs.",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(generate.router)


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    return [ModelInfo(key=k, huggingface_id=v) for k, v in MODELS.items()]
