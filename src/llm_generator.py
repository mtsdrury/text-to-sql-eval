"""Generate SQL from natural language using HuggingFace Inference API."""

import re
import time
from huggingface_hub import InferenceClient

from .schema import format_schema_for_prompt

MODELS = {
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

SYSTEM_PROMPT = """{schema}

You are a SQL expert. Given a natural language question about the Chinook database, write a SQLite-compatible SQL query that answers it.

Rules:
- Return ONLY the SQL query, no explanation or markdown formatting.
- Do not wrap the query in code blocks or backticks.
- You MUST use SQLite syntax. This is critical:
  - Use strftime('%Y', col) for year, strftime('%m', col) for month. There is NO QUARTER() function.
  - Use || for string concatenation. There is NO CONCAT() function.
  - Do NOT reference window function aliases in WHERE clauses. Use a subquery or CTE instead.
  - Example: instead of WHERE rk <= 3, wrap in a subquery: SELECT * FROM (SELECT ..., ROW_NUMBER() OVER (...) AS rk ...) WHERE rk <= 3
- Use ROUND(value, 2) on all calculated numeric columns (sums, averages, percentages).
- Format person names as FirstName || ' ' || LastName.
"""

USER_PROMPT = """Write a SQL query to answer the following question:

{question}"""


def build_prompt(question: str, schema_text: str | None = None) -> tuple[str, str]:
    """Build system and user prompts for the text-to-SQL task."""
    if schema_text is None:
        schema_text = format_schema_for_prompt()
    system = SYSTEM_PROMPT.format(schema=schema_text)
    user = USER_PROMPT.format(question=question)
    return system, user


def extract_sql(raw_output: str) -> str:
    """Extract SQL from model output, stripping markdown fences and explanations."""
    # First, try to extract content from code fences
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", raw_output, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip().rstrip(";").strip()

    # No code fences: extract SQL by finding SELECT/WITH and stopping at semicolons
    lines = raw_output.strip().split("\n")
    sql_lines = []
    in_sql = False
    for line in lines:
        stripped = line.strip().upper()
        if not in_sql and stripped.startswith(("SELECT", "WITH")):
            in_sql = True
        if in_sql:
            sql_lines.append(line)
            if stripped.endswith(";"):
                break

    result = "\n".join(sql_lines).strip().rstrip(";").strip()
    if not result:
        result = raw_output.strip().rstrip(";").strip()
    return result


def generate_sql(
    question: str,
    model_key: str,
    token: str,
    schema_text: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Generate SQL for a question using a specific model.

    Returns a dict with keys: model, question, generated_sql, raw_output, error, latency_s
    """
    model_id = MODELS[model_key]
    system_prompt, user_prompt = build_prompt(question, schema_text)
    client = InferenceClient(model=model_id, token=token)

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            latency = time.time() - start
            raw_output = response.choices[0].message.content
            generated_sql = extract_sql(raw_output)

            return {
                "model": model_key,
                "model_id": model_id,
                "question": question,
                "generated_sql": generated_sql,
                "raw_output": raw_output,
                "error": None,
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"  Retry {attempt + 1}/{max_retries} for {model_key} (waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                return {
                    "model": model_key,
                    "model_id": model_id,
                    "question": question,
                    "generated_sql": None,
                    "raw_output": None,
                    "error": str(e),
                    "latency_s": None,
                }
