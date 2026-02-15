"""POST /api/generate — single question SQL generation endpoint."""

import os

from fastapi import APIRouter, HTTPException

from src.llm_generator import generate_sql, MODELS
from src.query_runner import execute_query, compare_results
from src.schema import format_schema_for_prompt
from api.models import (
    GenerateRequest,
    GenerateResponse,
    ExecutionResult,
    ComparisonResult,
)
from api import wandb_logger

router = APIRouter(prefix="/api")

_schema_text: str | None = None


def _get_schema() -> str:
    global _schema_text
    if _schema_text is None:
        _schema_text = format_schema_for_prompt()
    return _schema_text


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {list(MODELS.keys())}",
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured on server")

    gen = generate_sql(req.question, req.model, token, _get_schema())

    if gen["error"]:
        wandb_logger.log_generation(
            model=req.model,
            question=req.question,
            latency_s=gen["latency_s"],
            execution_success=False,
        )
        return GenerateResponse(
            model=req.model,
            question=req.question,
            generated_sql=None,
            latency_s=gen["latency_s"],
            error=gen["error"],
        )

    execution = None
    comparison = None

    if req.execute and gen["generated_sql"]:
        exec_result = execute_query(gen["generated_sql"])
        execution = ExecutionResult(
            success=exec_result["success"],
            columns=exec_result["columns"],
            rows=[list(r) for r in exec_result["rows"]] if exec_result["rows"] else None,
            error=exec_result["error"],
        )

        if req.ground_truth_sql and exec_result["success"]:
            gt_result = execute_query(req.ground_truth_sql)
            if gt_result["success"]:
                cmp = compare_results(gt_result["rows"], exec_result["rows"])
                comparison = ComparisonResult(
                    exact_match=cmp["exact_match"],
                    fuzzy_match=cmp["fuzzy_match"],
                    row_count_match=cmp["row_count_match"],
                    row_overlap_pct=cmp["row_overlap_pct"],
                    gt_count=cmp["gt_count"],
                    gen_count=cmp["gen_count"],
                )

    wandb_logger.log_generation(
        model=req.model,
        question=req.question,
        latency_s=gen["latency_s"],
        execution_success=execution.success if execution else False,
        exact_match=comparison.exact_match if comparison else None,
    )

    return GenerateResponse(
        model=req.model,
        question=req.question,
        generated_sql=gen["generated_sql"],
        execution=execution,
        comparison=comparison,
        latency_s=gen["latency_s"],
    )
