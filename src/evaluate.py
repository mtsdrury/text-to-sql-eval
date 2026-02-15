"""Main evaluation script: run all queries x models, score, and save results."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .schema import format_schema_for_prompt
from .llm_generator import generate_sql, MODELS
from .query_runner import execute_query, compare_results

PROJECT_ROOT = Path(__file__).parent.parent
QUERIES_PATH = PROJECT_ROOT / "queries" / "ground_truth.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.json"
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env():
    """Load .env file into os.environ if it exists."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


_load_env()


def load_ground_truth() -> list[dict]:
    """Load ground truth queries from JSON."""
    with open(QUERIES_PATH) as f:
        return json.load(f)


def evaluate_single(
    query: dict,
    model_key: str,
    token: str,
    schema_text: str,
) -> dict:
    """Evaluate a single (model, query) pair."""
    query_id = query["id"]
    nl_question = query["nl_question"]
    gt_sql = query["sql"]

    print(f"  Query {query_id}: {nl_question[:60]}...")

    # Get ground truth results
    gt_result = execute_query(gt_sql)
    if not gt_result["success"]:
        print(f"    WARNING: Ground truth query failed: {gt_result['error']}")
        return {
            "query_id": query_id,
            "concept_tag": query["concept_tag"],
            "nl_question": nl_question,
            "model": model_key,
            "gt_error": gt_result["error"],
            "execution_success": False,
            "result_match": False,
            "failure_mode": "ground_truth_error",
        }

    # Generate SQL
    gen_result = generate_sql(nl_question, model_key, token, schema_text)

    if gen_result["error"]:
        print(f"    API error: {gen_result['error']}")
        return {
            "query_id": query_id,
            "concept_tag": query["concept_tag"],
            "nl_question": nl_question,
            "model": model_key,
            "generated_sql": None,
            "raw_output": gen_result["raw_output"],
            "execution_success": False,
            "result_match": False,
            "failure_mode": "api_error",
            "error_detail": gen_result["error"],
            "latency_s": gen_result["latency_s"],
        }

    generated_sql = gen_result["generated_sql"]

    # Execute generated SQL
    gen_exec = execute_query(generated_sql)

    if not gen_exec["success"]:
        print(f"    Execution failed: {gen_exec['error']}")
        return {
            "query_id": query_id,
            "concept_tag": query["concept_tag"],
            "nl_question": nl_question,
            "model": model_key,
            "generated_sql": generated_sql,
            "raw_output": gen_result["raw_output"],
            "execution_success": False,
            "result_match": False,
            "failure_mode": "execution_error",
            "error_detail": gen_exec["error"],
            "latency_s": gen_result["latency_s"],
        }

    # Compare results
    comparison = compare_results(gt_result["rows"], gen_exec["rows"])

    if comparison["exact_match"]:
        print(f"    EXACT MATCH ({comparison['gt_count']} rows)")
        failure_mode = None
    elif comparison["fuzzy_match"]:
        print(f"    FUZZY MATCH ({comparison['gt_count']} rows, numeric tolerance)")
        failure_mode = None
    else:
        if comparison["gen_count"] == 0:
            failure_mode = "empty_result"
        elif not comparison["row_count_match"]:
            failure_mode = "wrong_row_count"
        else:
            failure_mode = "wrong_values"
        overlap = comparison["row_overlap_pct"]
        print(f"    FAIL ({failure_mode}: expected {comparison['gt_count']} rows, "
              f"got {comparison['gen_count']}, {overlap}% overlap)")

    return {
        "query_id": query_id,
        "concept_tag": query["concept_tag"],
        "nl_question": nl_question,
        "model": model_key,
        "generated_sql": generated_sql,
        "raw_output": gen_result["raw_output"],
        "execution_success": True,
        "exact_match": comparison["exact_match"],
        "fuzzy_match": comparison["fuzzy_match"],
        "row_count_match": comparison["row_count_match"],
        "row_overlap_pct": comparison["row_overlap_pct"],
        "result_match": comparison["exact_match"] or comparison["fuzzy_match"],
        "failure_mode": failure_mode,
        "gt_row_count": comparison["gt_count"],
        "gen_row_count": comparison["gen_count"],
        "missing_count": comparison["missing_count"],
        "extra_count": comparison["extra_count"],
        "latency_s": gen_result["latency_s"],
    }


def run_evaluation(
    model_keys: list[str] | None = None,
    token: str | None = None,
) -> dict:
    """Run the full evaluation across all models and queries.

    Returns the full results dict (also saved to disk).
    """
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            print("ERROR: Set HF_TOKEN environment variable or pass token parameter.")
            sys.exit(1)

    if model_keys is None:
        model_keys = list(MODELS.keys())

    queries = load_ground_truth()
    schema_text = format_schema_for_prompt()

    print(f"Running evaluation: {len(model_keys)} models x {len(queries)} queries")
    print(f"Models: {', '.join(model_keys)}\n")

    # Load existing results to support resuming
    all_results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
        all_results = existing.get("results", [])
        existing_keys = {(r["model"], r["query_id"]) for r in all_results}
        print(f"Loaded {len(all_results)} existing results (resuming)\n")
    else:
        existing_keys = set()

    models_run = set()
    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({MODELS[model_key]})")
        print(f"{'='*60}")
        models_run.add(model_key)

        for query in queries:
            if (model_key, query["id"]) in existing_keys:
                print(f"  Query {query['id']}: (cached)")
                continue
            result = evaluate_single(query, model_key, token, schema_text)
            all_results.append(result)

            # Save incrementally after each query
            _save_results(all_results, model_keys, queries)

    output = _save_results(all_results, model_keys, queries)
    all_model_keys = list({r["model"] for r in all_results})
    summary = build_summary(all_results, all_model_keys)
    print_summary(summary, all_model_keys)

    return output


def _save_results(all_results: list[dict], model_keys: list[str], queries: list[dict]) -> dict:
    """Save current results to disk (called after each query for incremental saving)."""
    all_model_keys = list({r["model"] for r in all_results})
    summary = build_summary(all_results, all_model_keys)
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_models": len(all_model_keys),
            "num_queries": len(queries),
            "models": {k: MODELS[k] for k in all_model_keys if k in MODELS},
        },
        "summary": summary,
        "results": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    return output


def build_summary(results: list[dict], model_keys: list[str]) -> dict:
    """Build a summary table from evaluation results."""
    summary = {}
    for model_key in model_keys:
        model_results = [r for r in results if r["model"] == model_key]
        total = len(model_results)
        exec_success = sum(1 for r in model_results if r["execution_success"])
        exact_match = sum(1 for r in model_results if r.get("exact_match", r.get("result_match", False)))
        fuzzy_match = sum(1 for r in model_results if r.get("fuzzy_match", False))
        result_match = sum(1 for r in model_results if r.get("result_match", False))
        row_count_match = sum(1 for r in model_results if r.get("row_count_match", False))

        # Average row overlap for executed queries
        overlaps = [r["row_overlap_pct"] for r in model_results if r.get("row_overlap_pct") is not None]
        avg_overlap = round(sum(overlaps) / len(overlaps), 1) if overlaps else 0.0

        by_concept = {}
        for concept in ["basic", "intermediate", "window_function", "cte", "complex_combination"]:
            concept_results = [r for r in model_results if r["concept_tag"] == concept]
            by_concept[concept] = {
                "total": len(concept_results),
                "execution_success": sum(1 for r in concept_results if r["execution_success"]),
                "exact_match": sum(1 for r in concept_results if r.get("exact_match", r.get("result_match", False))),
                "fuzzy_match": sum(1 for r in concept_results if r.get("fuzzy_match", False)),
                "result_match": sum(1 for r in concept_results if r.get("result_match", False)),
            }

        failure_modes = {}
        for r in model_results:
            if r.get("failure_mode"):
                fm = r["failure_mode"]
                failure_modes[fm] = failure_modes.get(fm, 0) + 1

        summary[model_key] = {
            "total": total,
            "execution_success": exec_success,
            "execution_rate": round(exec_success / total * 100, 1) if total else 0,
            "row_count_match": row_count_match,
            "row_count_rate": round(row_count_match / total * 100, 1) if total else 0,
            "exact_match": exact_match,
            "exact_match_rate": round(exact_match / total * 100, 1) if total else 0,
            "fuzzy_match": fuzzy_match,
            "result_match": result_match,
            "accuracy": round(result_match / total * 100, 1) if total else 0,
            "avg_row_overlap_pct": avg_overlap,
            "by_concept": by_concept,
            "failure_modes": failure_modes,
        }

    return summary


def print_summary(summary: dict, model_keys: list[str]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")

    header = (f"{'Model':<20} {'Exec':>7} {'RowCt':>7} {'Fuzzy':>7} "
              f"{'Exact':>7} {'Overlap':>8} {'Total':>6}")
    print(header)
    print("-" * len(header))

    for model_key in model_keys:
        s = summary[model_key]
        print(
            f"{model_key:<20} {s['execution_rate']:>6.1f}% "
            f"{s['row_count_rate']:>6.1f}% "
            f"{s['accuracy']:>6.1f}% "
            f"{s['exact_match_rate']:>6.1f}% "
            f"{s['avg_row_overlap_pct']:>7.1f}% "
            f"{s['total']:>6}"
        )

    print(f"\n  Exec  = generated SQL runs without errors")
    print(f"  RowCt = correct number of result rows")
    print(f"  Fuzzy = exact or numeric-tolerance match")
    print(f"  Exact = strict set equality")
    print(f"  Overlap = avg % of ground-truth rows found (fuzzy)")

    print(f"\n{'By Difficulty (fuzzy match):'}")
    print(f"{'Model':<20} {'Basic':>8} {'Interm':>8} {'Window':>8} {'CTE':>8} {'Complex':>8}")
    print("-" * 60)

    for model_key in model_keys:
        bc = summary[model_key]["by_concept"]
        cols = []
        for concept in ["basic", "intermediate", "window_function", "cte", "complex_combination"]:
            c = bc.get(concept, {"result_match": 0, "total": 0})
            cols.append(f"{c['result_match']}/{c['total']}")
        print(f"{model_key:<20} " + "  ".join(f"{c:>6}" for c in cols))

    print(f"\n{'Failure Modes:'}")
    for model_key in model_keys:
        fm = summary[model_key]["failure_modes"]
        if fm:
            print(f"  {model_key}: {fm}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run text-to-SQL evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Models to evaluate (default: all)",
    )
    args = parser.parse_args()
    run_evaluation(model_keys=args.models)
