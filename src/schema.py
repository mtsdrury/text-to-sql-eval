"""Load and format the Chinook database schema for LLM prompts."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "chinook.db"


def get_create_statements(db_path: Path = DB_PATH) -> list[str]:
    """Extract CREATE TABLE statements from the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name;")
    statements = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return statements


def format_schema_for_prompt(db_path: Path = DB_PATH) -> str:
    """Format the full database schema as a string suitable for an LLM prompt."""
    statements = get_create_statements(db_path)
    schema_text = "\n\n".join(statements)
    return (
        "The following is the schema for a SQLite database called Chinook, "
        "which models a digital music store.\n\n"
        f"{schema_text}"
    )


if __name__ == "__main__":
    print(format_schema_for_prompt())
