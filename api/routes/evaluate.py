"""POST /api/evaluate — batch evaluation endpoint."""

import os

from fastapi import APIRouter, HTTPException

from src.llm_generator import MODELS
from src.schema import format_schema_for_prompt
from src.evaluate import evaluate_single, build_summary, load_ground_truth
from api.models import EvaluateRequest, EvaluateResponse, ModelSummary
from api import wandb_logger

router = APIRouter(prefix="/api")


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    for m in req.models:
        if m not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{m}'. Available: {list(MODELS.keys())}",
            )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured on server")

    queries = load_ground_truth()
    schema_text = format_schema_for_prompt()

    if req.query_ids:
        queries = [q for q in queries if q["id"] in req.query_ids]
    if req.concept_tags:
        queries = [q for q in queries if q["concept_tag"] in req.concept_tags]

    if not queries:
        raise HTTPException(status_code=400, detail="No queries match the provided filters")

    all_results = []
    for model_key in req.models:
        for query in queries:
            result = evaluate_single(query, model_key, token, schema_text)
            all_results.append(result)

    summary = build_summary(all_results, req.models)

    wandb_logger.log_evaluation(summary, all_results)

    response_summary = {}
    for model_key, stats in summary.items():
        response_summary[model_key] = ModelSummary(
            total=stats["total"],
            execution_rate=stats["execution_rate"],
            exact_match_rate=stats["exact_match_rate"],
            accuracy=stats["accuracy"],
            avg_row_overlap_pct=stats["avg_row_overlap_pct"],
            failure_modes=stats["failure_modes"],
        )

    return EvaluateResponse(
        summary=response_summary,
        wandb_run_url=wandb_logger.get_run_url(),
    )
