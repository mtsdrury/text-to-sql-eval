"""W&B logging helper — singleton run, graceful no-op if no API key."""

import os

_run = None


def init_run(project: str = "text-to-sql-eval") -> bool:
    """Initialize a W&B run. Returns True if W&B is active, False otherwise."""
    global _run
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb

        _run = wandb.init(project=project, reinit=True)
        return True
    except Exception as e:
        print(f"W&B init failed: {e}")
        _run = None
        return False


def is_enabled() -> bool:
    return _run is not None


def get_run_url() -> str | None:
    if _run is None:
        return None
    return _run.get_url()


def log_generation(
    model: str,
    question: str,
    latency_s: float | None,
    execution_success: bool,
    exact_match: bool | None = None,
) -> None:
    """Log a single generation request to W&B."""
    if _run is None:
        return
    metrics = {
        "model": model,
        "latency_s": latency_s or 0,
        "execution_success": int(execution_success),
    }
    if exact_match is not None:
        metrics["exact_match"] = int(exact_match)
    _run.log(metrics)


def log_evaluation(summary: dict, results: list[dict]) -> None:
    """Log a batch evaluation to W&B as a table."""
    if _run is None:
        return
    try:
        import wandb

        columns = [
            "model", "query_id", "concept_tag", "execution_success",
            "exact_match", "latency_s",
        ]
        table = wandb.Table(columns=columns)
        for r in results:
            table.add_data(
                r.get("model", ""),
                r.get("query_id", ""),
                r.get("concept_tag", ""),
                r.get("execution_success", False),
                r.get("result_match", False),
                r.get("latency_s", 0),
            )
        _run.log({"evaluation_results": table})

        for model_key, stats in summary.items():
            _run.log({
                f"{model_key}/execution_rate": stats["execution_rate"],
                f"{model_key}/accuracy": stats["accuracy"],
                f"{model_key}/exact_match_rate": stats["exact_match_rate"],
            })
    except Exception as e:
        print(f"W&B evaluation logging failed: {e}")


def finish() -> None:
    """Finish the W&B run."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None
