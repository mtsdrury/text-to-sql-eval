"""Execute SQL queries against the Chinook database and compare results."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "chinook.db"


def execute_query(sql: str, db_path: Path = DB_PATH) -> dict:
    """Execute a SQL query and return results or error.

    Returns a dict with keys: success, rows, columns, error
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        return {
            "success": True,
            "rows": rows,
            "columns": columns,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "rows": None,
            "columns": None,
            "error": str(e),
        }


def _normalize_value(val) -> str:
    """Normalize a single value to a string for comparison."""
    if val is None:
        return "NULL"
    if isinstance(val, float):
        return str(round(val, 2))
    return str(val)


def _fuzzy_match_value(a, b, tolerance: float = 0.01) -> bool:
    """Check if two values match, with numeric tolerance."""
    if a == b:
        return True
    # Try numeric comparison with tolerance
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tolerance
    except (ValueError, TypeError):
        return False


def _normalize_row(row: tuple) -> tuple:
    """Normalize a row's values for comparison."""
    return tuple(_normalize_value(v) for v in row)


def _fuzzy_match_row(row_a: tuple, row_b: tuple, tolerance: float = 0.01) -> bool:
    """Check if two normalized rows match with numeric tolerance."""
    if len(row_a) != len(row_b):
        return False
    return all(_fuzzy_match_value(a, b, tolerance) for a, b in zip(row_a, row_b))


def compare_results(ground_truth_rows: list[tuple], generated_rows: list[tuple]) -> dict:
    """Compare two result sets with both exact and fuzzy matching.

    Returns a dict with:
      - exact_match: bool, strict set equality
      - fuzzy_match: bool, all rows match with numeric tolerance
      - row_count_match: bool, same number of rows
      - row_overlap_pct: float, % of GT rows found in generated (fuzzy)
      - gt_count, gen_count, missing_count, extra_count
    """
    gt_norm = [_normalize_row(r) for r in ground_truth_rows]
    gen_norm = [_normalize_row(r) for r in generated_rows]

    gt_set = set(gt_norm)
    gen_set = set(gen_norm)

    # Exact match (strict set equality)
    exact_match = gt_set == gen_set

    # Row count match
    row_count_match = len(gt_set) == len(gen_set)

    # Fuzzy matching: for each GT row, find a fuzzy match in generated
    gen_remaining = list(gen_norm)
    fuzzy_matched = 0
    for gt_row in gt_norm:
        for i, gen_row in enumerate(gen_remaining):
            if _fuzzy_match_row(gt_row, gen_row):
                fuzzy_matched += 1
                gen_remaining.pop(i)
                break

    fuzzy_match = fuzzy_matched == len(gt_norm) and len(gen_norm) == len(gt_norm)
    row_overlap_pct = round(fuzzy_matched / len(gt_norm) * 100, 1) if gt_norm else 0.0

    missing = gt_set - gen_set
    extra = gen_set - gt_set

    return {
        "exact_match": exact_match,
        "fuzzy_match": fuzzy_match,
        "row_count_match": row_count_match,
        "row_overlap_pct": row_overlap_pct,
        "gt_count": len(gt_set),
        "gen_count": len(gen_set),
        "missing_count": len(missing),
        "extra_count": len(extra),
    }
