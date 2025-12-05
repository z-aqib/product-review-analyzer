import csv
import datetime
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from gradio_client import Client

# Import your existing ML, RAG, and merge logic
# NOTE: these imports assume this file lives in src/llm/experiments/
# and that src is a package. Run via:
#   python -m src.llm.experiments.run_experiments
from ...ml.service import get_ml_candidates_for_user
from ...rag.rag_service import ask
from ..advisor import merge_ml_and_rag


# ================================
# 1. Paths & constants
# ================================

THIS_DIR = Path(__file__).resolve().parent

CONFIG_PATH = THIS_DIR / "experiments_config.csv"
RESULTS_PATH = THIS_DIR / "experiment_results.csv"
SAMPLE_RESPONSES_PATH = THIS_DIR / "sample_responses.json"

# Default user_id if none provided in CSV
DEFAULT_USER_ID = os.getenv(
    "EXPERIMENT_DEFAULT_USER_ID",
    "AG3D6O4STAQKAY2UVGEUV46KN35Q",
)

# Hugging Face Space settings (Haaris' model)
SPACE_ID = os.getenv("QWEN_SPACE_ID", "MuhammadHaaris/mlops")
SPACE_API_NAME = os.getenv("QWEN_API_NAME", "/predict")

# Single shared client for all calls
_qwen_client = Client(SPACE_ID)


# ================================
# 2. Helpers: HF Space + context
# ================================


def call_space(
    final_prompt: str, max_retries: int = 3, backoff_seconds: float = 5.0
) -> Tuple[str, float]:
    """
    Send final prompt to the Hugging Face Space (Qwen SFT model)
    and return (response_text, latency_seconds).

    We add simple retry logic because the Space or network can be flaky
    (e.g., SSL handshake timeouts).
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        print(f"        [call_space] Attempt {attempt}/{max_retries}...")
        start = time.time()
        try:
            result = _qwen_client.predict(
                user_input=final_prompt,
                api_name=SPACE_API_NAME,
            )
            elapsed = time.time() - start
            print(f"        [call_space] ✅ Success on attempt {attempt}, latency={elapsed:.3f}s")
            return str(result), elapsed
        except Exception as e:
            elapsed = time.time() - start
            last_error = e
            print(f"        [call_space] ❌ Failed on attempt {attempt} after {elapsed:.3f}s")
            print(f"        [call_space] Error: {e}")

            if attempt < max_retries:
                print(f"        [call_space] Retrying in {backoff_seconds} seconds...")
                time.sleep(backoff_seconds)

    # If we reach here, all attempts failed; raise the last error so main() can catch it
    print("        [call_space] ❌ All attempts failed. Giving up for this experiment.")
    raise last_error if last_error is not None else RuntimeError("Unknown error calling Space")


def build_context_block(
    user_query: str,
    ml_candidates: List[Dict],
    rag_result: Dict,
) -> str:
    """
    Build a text block that includes:
    - User query
    - Merged product candidates (ML + RAG)
    - RAG review summary

    This is what gets appended to llm_prompt to form final_prompt.
    """

    merged_products = merge_ml_and_rag(
        ml_candidates=ml_candidates,
        rag_products=rag_result.get("products", []),
    )

    lines: List[str] = []

    # User query
    lines.append(f"User query: {user_query.strip()}")
    lines.append("")

    # Products block
    if merged_products:
        lines.append("Product candidates (combined from ML ranking and RAG):")
        for idx, p in enumerate(merged_products, start=1):
            name = p.get("product_name") or f"Product {idx}"
            pid = p.get("product_id")
            price = p.get("price")
            rating = p.get("rating")
            ml_score = p.get("ml_score")
            retrieval_score = p.get("retrieval_score")
            snippet = (p.get("snippet") or "")[:300]

            detail_bits = []
            if price is not None:
                detail_bits.append(f"price ≈ {price}")
            if rating is not None:
                detail_bits.append(f"rating ≈ {rating}")
            if ml_score is not None:
                detail_bits.append(f"ml_score ≈ {ml_score:.3f}")
            if retrieval_score is not None:
                detail_bits.append(f"retrieval_score ≈ {retrieval_score:.3f}")

            if pid:
                base_line = f"{idx}. {name} (ID: {pid})"
            else:
                base_line = f"{idx}. {name}"

            if detail_bits:
                base_line += " — " + ", ".join(detail_bits)

            lines.append(base_line)

            if snippet:
                lines.append(f"   Reviews snippet: {snippet}")
        lines.append("")

    # RAG summary
    rag_answer = rag_result.get("rag_answer") or rag_result.get("answer") or ""
    if rag_answer.strip():
        lines.append("RAG review summary:")
        lines.append(rag_answer.strip())

    return "\n".join(lines).strip()


# ================================
# 3. Sample responses (few-shot)
# ================================


def load_sample_responses() -> List[Dict]:
    """
    Load query/response/review items from sample_responses.json.
    """
    if not SAMPLE_RESPONSES_PATH.exists():
        print(f"⚠️ sample_responses.json not found at {SAMPLE_RESPONSES_PATH}")
        return []

    with SAMPLE_RESPONSES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])
    print(f"✅ Loaded {len(items)} sample responses from {SAMPLE_RESPONSES_PATH}")
    return items


def select_examples(
    sample_items: List[Dict],
    k: int,
    sample_type: str | None = None,
    positive_only: bool = True,
) -> List[Dict]:
    """
    Select up to k examples, optionally filtering by type and review sentiment.
    """
    if k <= 0 or not sample_items:
        return []

    filtered: List[Dict] = []

    for item in sample_items:
        # Match on type (Earphones / Headphones, TVs, Phones, etc.)
        if sample_type and item.get("type") != sample_type:
            continue

        if positive_only:
            review = (item.get("review_of_response") or "").lower()
            # Skip clearly bad ones
            if "not very good" in review or "worst" in review:
                continue

        filtered.append(item)

    # If not enough in this type, fall back to global pool (still positive_only)
    if len(filtered) < k and sample_type:
        for item in sample_items:
            if item in filtered:
                continue
            if positive_only:
                review = (item.get("review_of_response") or "").lower()
                if "not very good" in review or "worst" in review:
                    continue
            filtered.append(item)

    return filtered[:k]


def format_examples_for_prompt(examples: List[Dict]) -> str:
    """
    Turn selected examples into a string block for few-shot prompting.
    """
    lines: List[str] = []
    for idx, ex in enumerate(examples, start=1):
        q = (ex.get("query") or "").strip()
        r = (ex.get("response") or "").strip()

        lines.append(f"Example {idx}:")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {r}")
        lines.append("")  # blank line between examples

    return "\n".join(lines).strip()


def build_llm_prompt_for_strategy(
    base_prompt: str,
    strategy: str,
    few_shot_k: int,
    sample_type: str | None,
    user_query: str,
    sample_items: List[Dict],
) -> str:
    """
    Build the final LLM *instruction* prompt (before ML + RAG context)
    based on:
      - strategy (zero_shot, few_shot_3, few_shot_5, cot, meta, ...)
      - few_shot_k
      - sample_type (TVs, Phones, etc.)
      - sample_responses examples.
    """
    strategy = (strategy or "zero_shot").lower()
    base_prompt = (base_prompt or "").strip()

    if not base_prompt:
        base_prompt = (
            "You are an expert product review advisor. "
            "Use the ML ranking and RAG snippets (which will be provided after this prompt) "
            "to recommend the best product(s) for the user. "
            "Be specific, mention trade-offs, and stay within the given products."
        )

    prompt = base_prompt

    # Decide k if not explicitly set
    k = few_shot_k or (3 if "3" in strategy else 5 if "5" in strategy else 0)

    # Few-shot strategies
    if "few_shot" in strategy or k > 0:
        examples = select_examples(
            sample_items=sample_items,
            k=k,
            sample_type=sample_type,
            positive_only=True,
        )
        if examples:
            examples_block = format_examples_for_prompt(examples)
            prompt += (
                "\n\nHere are example conversations between a user and the assistant. "
                "Follow the helpful, structured style shown in these examples:\n\n"
                f"{examples_block}\n\n"
                "Now answer the *next* user query using the ML + RAG context that will be "
                "provided after this prompt."
            )

    # Advanced: chain-of-thought (CoT)
    if strategy == "cot":
        prompt += (
            "\n\nWhen answering, first think step-by-step about the options using the ML and "
            "RAG information. Explicitly compare the top candidates. Then end with a short "
            "section titled 'Final Recommendation' with 2–4 sentences."
        )

    # Advanced: meta-prompting
    if strategy == "meta":
        prompt += (
            "\n\nYou are a brutally honest product advisor. You must:\n"
            "- Pick at most 1–2 main options, not a long list.\n"
            "- Clearly list pros and cons based on reviews.\n"
            "- Call out if information is missing or uncertain.\n\n"
            "Output format:\n"
            "1. Short answer (1–2 sentences)\n"
            "2. Bullet list of pros and cons for each recommended product\n"
            "3. Final recommendation explaining who this is best for."
        )

    return prompt


# ================================
# 4. Config + results helpers
# ================================


def load_config_rows() -> List[Dict[str, str]]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_existing_results_ids() -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Returns:
      - list of fieldnames (if file exists, else empty)
      - list of existing result rows
    """
    if not RESULTS_PATH.exists():
        return [], []

    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


# ================================
# 5. Main experiment runner
# ================================


def main() -> None:
    """
    Main entry-point for running all LLM experiments.

    Flow:
      1) Load experiments_config.csv rows.
      2) Load sample_responses.json for few-shot examples.
      3) Load experiment_results.csv (if exists) to see which experiment_ids are done.
      4) For each config row:
         - Skip if experiment_id is empty or already done.
         - Build user_query (variant or original).
         - Decide user_id (from CSV or default).
         - Build llm_prompt for this strategy (zero-shot / few-shot / CoT / meta).
         - Run ML + RAG using your existing services.
         - Build a context block from ML + RAG outputs.
         - Combine llm_prompt + context into final_prompt.
         - Call Hugging Face Space (Qwen SFT model).
         - Log timings + response into experiment_results.csv.
    """

    print("==============================================")
    print("🚀 Starting LLM experiments runner")
    print("==============================================")
    print(f"Config file path   : {CONFIG_PATH}")
    print(f"Results file path  : {RESULTS_PATH}")
    print(f"Sample responses   : {SAMPLE_RESPONSES_PATH}")
    print(f"Default user_id    : {DEFAULT_USER_ID}")
    print(f"HuggingFace Space  : {SPACE_ID} (api_name={SPACE_API_NAME})")
    print("==============================================\n")

    # 1) Load configs
    print("[STEP 1] Loading experiment configuration rows...")
    config_rows = load_config_rows()
    if not config_rows:
        print(f"❌ No rows found in {CONFIG_PATH}. Add experiments first.")
        return

    config_fieldnames = list(config_rows[0].keys())
    print(f"✅ Loaded {len(config_rows)} experiment rows from config.")
    print(f"   Config columns: {config_fieldnames}\n")

    # 2) Load sample responses
    print("[STEP 2] Loading sample responses for few-shot prompting...")
    sample_items = load_sample_responses()
    print()

    # 3) Load existing results to avoid duplicates
    print("[STEP 3] Loading existing results (if any) to avoid duplicate runs...")
    existing_fieldnames, existing_rows = load_existing_results_ids()
    done_ids = {row.get("experiment_id") for row in existing_rows if row.get("experiment_id")}

    if existing_rows:
        print(f"✅ Found existing results with {len(existing_rows)} rows.")
        print(f"   Example existing experiment_ids (up to 5): {list(done_ids)[:5]}")
    else:
        print("ℹ️ No existing results file found; all experiments will be treated as new.")
    print()

    # 4) Prepare result fieldnames
    print("[STEP 4] Preparing output result schema...")
    extra_fields = [
        "start_time_iso",
        "end_time_iso",
        "latency_seconds",
        "response_chars",
        "response_text",
        "error",
    ]

    result_fieldnames = config_fieldnames + extra_fields
    print(f"   Result columns will be: {result_fieldnames}\n")

    # 5) Open results file for append, write header if new
    print("[STEP 5] Opening results file for append...")
    file_exists = RESULTS_PATH.exists()
    out_file = RESULTS_PATH.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=result_fieldnames)

    if not file_exists:
        print("   Results file does not exist yet. Writing header row...")
        writer.writeheader()
    else:
        print("   Results file already exists. Appending new rows without rewriting header.")
    print()

    # 6) Loop over experiments
    print("[STEP 6] Iterating over experiment rows...\n")
    new_runs = 0
    total_experiments = len(config_rows)

    for idx, row in enumerate(config_rows, start=1):
        print("--------------------------------------------------")
        print(f"▶ Experiment row {idx}/{total_experiments}")

        exp_id = (row.get("experiment_id") or "").strip()
        if not exp_id:
            print("   ⚠️ Skipping row because experiment_id is missing/empty.")
            continue

        print(f"   experiment_id   : {exp_id}")

        if exp_id in done_ids:
            print(
                "   ℹ️ This experiment_id is already present in results. Skipping to avoid duplicate."
            )
            continue

        # Strategy, sample_type, few_shot_k
        strategy = (row.get("strategy") or "zero_shot").strip()
        sample_type = (row.get("sample_type") or "").strip() or None
        few_shot_k_str = (row.get("few_shot_k") or "").strip()
        few_shot_k = int(few_shot_k_str) if few_shot_k_str.isdigit() else 0

        print(f"   strategy        : {strategy}")
        print(f"   sample_type     : {sample_type}")
        print(f"   few_shot_k      : {few_shot_k}")

        # Choose user query: prefer variant, fallback to original
        user_query_variant = (row.get("user_query_variant") or "").strip()
        user_query_original = (row.get("user_query_original") or "").strip()
        user_query = user_query_variant or user_query_original

        if not user_query:
            print(f"   ⚠️ Skipping {exp_id}: no user_query_variant or user_query_original found.")
            continue

        print(f"   user_query      : {user_query!r}")

        # Choose user_id (optional column). If not present or empty, use DEFAULT_USER_ID.
        user_id = (row.get("user_id") or "").strip() or DEFAULT_USER_ID
        print(f"   user_id         : {user_id}")

        # Base LLM prompt (instructions, examples, etc.)
        base_llm_prompt = (row.get("llm_prompt") or "").rstrip()
        print("   base llm_prompt set? : ", "YES" if base_llm_prompt else "NO (using default)")

        # Build final instruction prompt for this strategy
        llm_prompt = build_llm_prompt_for_strategy(
            base_prompt=base_llm_prompt,
            strategy=strategy,
            few_shot_k=few_shot_k,
            sample_type=sample_type,
            user_query=user_query,
            sample_items=sample_items,
        )

        # 6a) Run ML + RAG
        print("   [6a] Running ML recommender...")
        try:
            ml_candidates = get_ml_candidates_for_user(user_id, user_query)
        except TypeError:
            print("        ⚠️ get_ml_candidates_for_user(user_id, user_query) failed (TypeError).")
            print("        → Retrying with get_ml_candidates_for_user(user_id) only.")
            ml_candidates = get_ml_candidates_for_user(user_id)

        print(f"        ✅ ML returned {len(ml_candidates)} candidates.")

        print("   [6a] Running RAG...")
        try:
            rag_result = ask(user_query, k=5)
        except TypeError:
            print("        ⚠️ ask(user_query, k=5) failed (TypeError).")
            print("        → Retrying with ask(user_query) only.")
            rag_result = ask(user_query)

        if isinstance(rag_result, dict):
            rag_products = rag_result.get("products") or rag_result.get("items") or []
            print(f"        ✅ RAG returned {len(rag_products)} product items.")
        else:
            print("        ⚠️ RAG result is not a dict. Type:", type(rag_result))

        # 6b) Build context block and final prompt
        print("   [6b] Building context block from ML + RAG outputs...")
        context_block = build_context_block(
            user_query=user_query,
            ml_candidates=ml_candidates,
            rag_result=rag_result,
        )
        print("        ✅ Context block built.")
        print("        (First 200 chars of context):")
        print(
            "        ",
            context_block[:200].replace("\n", " ") + ("..." if len(context_block) > 200 else ""),
        )
        print()

        final_prompt = f"{llm_prompt}\n\n{context_block}"

        print("   Preview of final_prompt (first 300 chars):")
        print(
            "   ",
            final_prompt[:300].replace("\n", " ") + ("..." if len(final_prompt) > 300 else ""),
        )
        print()

        # 6c) Call HF Space
        print("   [6c] Calling Hugging Face Space (Qwen SFT model)...")
        start_time = datetime.datetime.now()
        start_iso = start_time.isoformat(timespec="seconds")
        print(f"        Start time: {start_iso}")

        response_text = ""
        latency_seconds = None
        error_text = ""

        try:
            response_text, latency_seconds = call_space(final_prompt)
            print("        ✅ Space call succeeded.")
            print(f"        Latency    : {latency_seconds:.3f} seconds")
            print("        Response preview (first 200 chars):")
            print(
                "        ",
                response_text[:200].replace("\n", " ")
                + ("..." if len(response_text) > 200 else ""),
            )
        except Exception as e:
            error_text = str(e)
            print("        ❌ Space call FAILED with error:")
            print("           ", error_text)

        end_time = datetime.datetime.now()
        end_iso = end_time.isoformat(timespec="seconds")
        print(f"        End time   : {end_iso}")

        # 6d) Build result row
        print("   [6d] Writing result row to CSV...")
        result_row: Dict[str, str] = {}

        for key in config_fieldnames:
            result_row[key] = row.get(key, "")

        result_row["start_time_iso"] = start_iso
        result_row["end_time_iso"] = end_iso
        result_row["latency_seconds"] = (
            f"{latency_seconds:.3f}" if latency_seconds is not None else ""
        )
        result_row["response_chars"] = str(len(response_text))
        result_row["response_text"] = response_text
        result_row["error"] = error_text

        writer.writerow(result_row)
        out_file.flush()
        new_runs += 1

        print(f"   ✅ Completed experiment {exp_id} (error={bool(error_text)})")
        print("--------------------------------------------------\n")

    out_file.close()
    print("==============================================")
    print(f"🏁 All done. New experiments run in this session: {new_runs}")
    print(f"📄 Results saved in: {RESULTS_PATH}")
    print("==============================================")


if __name__ == "__main__":
    main()
