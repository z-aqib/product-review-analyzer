# src/llm/experiments/run_experiments.py
import csv
import datetime
import json
import os
import time
import re  # === NEW (D1) ===
from pathlib import Path
from typing import Dict, List, Tuple

from gradio_client import Client

# === NEW (D1) ===
# Use MLflow to log quantitative metrics as required by D1.
try:
    import mlflow
except ImportError:
    mlflow = None

# Import your existing ML, RAG, and merge logic
from ...ml.service import get_ml_candidates_for_user
from ...rag.rag_service import ask
from ..advisor import merge_ml_and_rag


# ================================
# 1. Paths & constants
# ================================

THIS_DIR = Path(__file__).resolve().parent

# Default config (full experiments)
DEFAULT_CONFIG_PATH = THIS_DIR / "experiments_config.csv"

# CI-specific config (small subset so GitHub Actions stays fast + cheap)
CI_CONFIG_PATH = THIS_DIR / "experiments_config_ci.csv"

# If running inside CI (GitHub Actions sets CI=true), prefer the CI config
if os.getenv("CI", "").lower() == "true" and CI_CONFIG_PATH.exists():
    CONFIG_PATH = CI_CONFIG_PATH
else:
    CONFIG_PATH = DEFAULT_CONFIG_PATH

RESULTS_PATH = THIS_DIR / "experiment_results.csv"
SAMPLE_RESPONSES_PATH = THIS_DIR / "sample_responses.json"

# === NEW (D1) ===
# Try to load eval.jsonl from data/eval.jsonl (spec requirement),
# but fall back to experiments/ if needed.
PROJECT_ROOT = THIS_DIR.parents[2]
EVAL_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "eval.jsonl",
    THIS_DIR / "eval.jsonl",
]

# Default user_id if none provided in CSV
DEFAULT_USER_ID = os.getenv(
    "EXPERIMENT_DEFAULT_USER_ID",
    "AG3D6O4STAQKAY2UVGEUV46KN35Q",
)

# Hugging Face Space settings (Haaris' model)
SPACE_ID = os.getenv("QWEN_SPACE_ID", "MuhammadHaaris/mlops")
SPACE_API_NAME = os.getenv("QWEN_API_NAME", "/predict")

# Single shared client for all calls
# We also configure HTTP-level timeouts via httpx_kwargs so that low-level
# connections don't hang forever.
_qwen_client = Client(
    SPACE_ID,
    httpx_kwargs={
        "timeout": 610.0,  # seconds; slightly above our per-attempt job timeout
    },
)

# ================================
# 2. Helpers: HF Space + context
# ================================


def call_space(
    final_prompt: str,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
    per_attempt_timeout: float = 600.0,
) -> Tuple[str, float]:
    """
    Send final prompt to the Hugging Face Space (Qwen SFT model)
    and return (response_text, latency_seconds).

    Improvements added:
    - Use Client.submit() + Job.result(timeout=...) to enforce a HARD timeout
      per attempt (default 600s = 10 minutes).
    - Retry transient failures up to `max_retries` times with a fixed backoff.
    - Never let a single call hang for hours.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        print(f"        [call_space] Attempt {attempt}/{max_retries}...")
        start = time.time()
        try:
            # Run the remote prediction in a background job
            job = _qwen_client.submit(
                user_input=final_prompt,
                api_name=SPACE_API_NAME,
            )

            # Block until result or timeout
            result = job.result(timeout=per_attempt_timeout)

            elapsed = time.time() - start
            print(
                f"        [call_space] ✅ Success on attempt {attempt}, " f"latency={elapsed:.3f}s"
            )
            return str(result), elapsed

        except TimeoutError as e:
            # Job didn't finish within `per_attempt_timeout`
            elapsed = time.time() - start
            last_error = e
            print(
                f"        [call_space] ❌ Timeout on attempt {attempt} "
                f"after {elapsed:.3f}s (>{per_attempt_timeout}s)."
            )
            print(f"        [call_space] Error: {e}")

        except Exception as e:
            # Any other network / client / HF error
            elapsed = time.time() - start
            last_error = e
            print(f"        [call_space] ❌ Failed on attempt {attempt} after {elapsed:.3f}s")
            print(f"        [call_space] Error: {e}")

        # If we reach here, this attempt failed. Decide whether to retry.
        if attempt < max_retries:
            print(f"        [call_space] Retrying in {backoff_seconds} seconds...")
            time.sleep(backoff_seconds)

    # After exhausting retries, raise the last seen error
    print("        [call_space] ❌ All attempts failed. Giving up for this experiment.")
    raise (last_error if last_error is not None else RuntimeError("Unknown error calling Space"))


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

    rag_answer = rag_result.get("rag_answer") or rag_result.get("answer") or ""
    if rag_answer.strip():
        lines.append("RAG review summary:")
        lines.append(rag_answer.strip())

    return "\n".join(lines).strip()


# ================================
# 3. Sample responses (few-shot)
# ================================


def load_sample_responses() -> List[Dict]:
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
    if k <= 0 or not sample_items:
        return []

    filtered: List[Dict] = []

    for item in sample_items:
        if sample_type and item.get("type") != sample_type:
            continue

        if positive_only:
            review = (item.get("review_of_response") or "").lower()
            if "not very good" in review or "worst" in review:
                continue

        filtered.append(item)

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
    lines: List[str] = []
    for idx, ex in enumerate(examples, start=1):
        q = (ex.get("query") or "").strip()
        r = (ex.get("response") or "").strip()

        lines.append(f"Example {idx}:")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {r}")
        lines.append("")

    return "\n".join(lines).strip()


def build_llm_prompt_for_strategy(
    base_prompt: str,
    strategy: str,
    few_shot_k: int,
    sample_type: str | None,
    user_query: str,
    sample_items: List[Dict],
) -> str:
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

    k = few_shot_k or (3 if "3" in strategy else 5 if "5" in strategy else 0)

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

    if strategy == "cot":
        prompt += (
            "\n\nWhen answering, first think step-by-step about the options using the ML and "
            "RAG information. Explicitly compare the top candidates. Then end with a short "
            "section titled 'Final Recommendation' with 2–4 sentences."
        )

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
# 4. Config + eval + results helpers
# ================================


def load_config_rows() -> List[Dict[str, str]]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_existing_results_ids(
    expected_fieldnames: List[str],
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Load existing results in a way that works for BOTH:
    - legacy files without a header row
    - newer files with a proper header

    Strategy:
    - Read raw rows with csv.reader.
    - If the first row clearly looks like a header (contains 'experiment_id'
      or all cells are within expected_fieldnames), treat it as header.
    - Otherwise treat the file as headerless and map columns positionally
      using expected_fieldnames.
    """
    if not RESULTS_PATH.exists():
        return [], []

    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        # Empty file – treat as no data yet.
        return expected_fieldnames, []

    header_candidate = raw_rows[0]

    # Heuristics to decide if first row is a header
    has_experiment_header = "experiment_id" in header_candidate
    all_in_expected = all(cell in expected_fieldnames for cell in header_candidate)

    if has_experiment_header or all_in_expected:
        # Looks like a proper header
        fieldnames = header_candidate
        data_rows = raw_rows[1:]
    else:
        # Legacy file without header: assume columns follow expected_fieldnames
        fieldnames = expected_fieldnames
        data_rows = raw_rows

    dict_rows: List[Dict[str, str]] = []
    for row in data_rows:
        row_dict: Dict[str, str] = {}
        for i, name in enumerate(fieldnames):
            row_dict[name] = row[i] if i < len(row) else ""
        dict_rows.append(row_dict)

    return fieldnames, dict_rows


# === NEW (D1): Load eval.jsonl as scenario_id → record ===
def load_eval_dataset() -> Dict[str, Dict[str, str]]:
    eval_data: Dict[str, Dict[str, str]] = {}
    chosen_path = None

    for path in EVAL_PATH_CANDIDATES:
        if path.exists():
            chosen_path = path
            break

    if not chosen_path:
        print("⚠️ No eval.jsonl found in any candidate path. Quantitative metrics will be empty.")
        return eval_data

    print(f"✅ Loading eval dataset from: {chosen_path}")
    with chosen_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("   ⚠️ Skipping invalid JSON line in eval.jsonl:", line[:80])
                continue

            scenario_id = (obj.get("scenario_id") or "").strip()
            if not scenario_id:
                print("   ⚠️ Skipping eval line without scenario_id.")
                continue

            eval_data[scenario_id] = obj

    print(f"✅ Loaded {len(eval_data)} eval scenarios.\n")
    return eval_data


# === NEW (D1): Simple token-overlap F1 metric ===
def _normalize_text_for_metric(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def token_overlap_f1(ref: str, pred: str) -> float:
    ref_tokens = _normalize_text_for_metric(ref)
    pred_tokens = _normalize_text_for_metric(pred)

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    pred_set = set(pred_tokens)

    inter = len(ref_set & pred_set)
    prec = inter / len(pred_set)
    rec = inter / len(ref_set)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ================================
# 5. Main experiment runner
# ================================


def main() -> None:
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

    # === NEW (D1) ===
    # 2.5) Load eval dataset
    print("[STEP 2.5] Loading eval.jsonl for quantitative metrics...")
    eval_map = load_eval_dataset()

    # 3) Prepare result fieldnames / schema
    print("[STEP 3] Preparing output result schema...")
    extra_fields = [
        "start_time_iso",
        "end_time_iso",
        "latency_seconds",
        "response_chars",
        "response_text",
        "error",
        # D1 fields:
        "scenario_id",  # will be skipped here if already in config_fieldnames
        "ideal_answer",
        "metric_token_overlap_f1",
        "helpfulness_score",
        "factuality_score",
    ]

    result_fieldnames = config_fieldnames + [f for f in extra_fields if f not in config_fieldnames]
    print(f"   Result columns will be: {result_fieldnames}\n")

    # 4) Load existing results (if any) using robust header / headerless loader
    print("[STEP 4] Loading existing results (if any) to avoid duplicate runs...")
    existing_fieldnames, existing_rows = load_existing_results_ids(result_fieldnames)

    done_ids = {row.get("experiment_id") for row in existing_rows if row.get("experiment_id")}

    if existing_rows:
        print(f"✅ Found existing results with {len(existing_rows)} rows.")
        print(f"   Example existing experiment_ids (up to 5): {list(done_ids)[:5]}")
        if "experiment_id" not in existing_fieldnames:
            print(
                "   ⚠️ Existing file appears to be legacy (no 'experiment_id' header).\n"
                "      Duplicate skipping is still enabled using positional mapping,\n"
                "      but consider adding a header row for better readability."
            )
    else:
        print("ℹ️ No existing results file found; all experiments will be treated as new.")
    print()

    # 5) Open results file for append, write header if file is new
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

    # === NEW (D1): Initialize MLflow experiment ===
    if mlflow is not None:
        mlflow.set_experiment("llm_prompt_eval")

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

        # Scenario ID (for eval.jsonl lookup)
        scenario_id = (row.get("scenario_id") or "").strip()
        print(f"   scenario_id     : {scenario_id or '(none)'}")

        # Choose user query
        user_query_variant = (row.get("user_query_variant") or "").strip()
        user_query_original = (row.get("user_query_original") or "").strip()
        user_query = user_query_variant or user_query_original

        if not user_query:
            print(f"   ⚠️ Skipping {exp_id}: no user_query_variant or user_query_original found.")
            continue

        print(f"   user_query      : {user_query!r}")

        # User ID
        user_id = (row.get("user_id") or "").strip() or DEFAULT_USER_ID
        print(f"   user_id         : {user_id}")

        # Base LLM prompt
        base_llm_prompt = (row.get("llm_prompt") or "").rstrip()
        print(
            "   base llm_prompt set? : ",
            "YES" if base_llm_prompt else "NO (using default)",
        )

        # Build final instruction prompt
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

        rag_result = {}
        rag_error_text = ""

        try:
            try:
                # Preferred call: use top-k retrieval
                rag_result = ask(user_query, k=5)
            except TypeError:
                # Backwards compatibility with older signatures
                print("        ⚠️ ask(user_query, k=5) failed (TypeError).")
                print("        → Retrying with ask(user_query) only.")
                rag_result = ask(user_query)
        except Exception as e:
            # Catch *any* RAG/Gemini failure (DNS, timeout, gRPC, etc.)
            rag_error_text = str(e)
            print("        ❌ RAG call FAILED.")
            print("           Error:", rag_error_text)
            # Fall back to an empty result so we can still build a context block
            rag_result = {
                "products": [],
                "rag_answer": "",
            }

        if isinstance(rag_result, dict):
            rag_products = rag_result.get("products") or rag_result.get("items") or []
            print(f"        ✅ RAG returned {len(rag_products)} product items.")
        else:
            print("        ⚠️ RAG result is not a dict. Type:", type(rag_result))
            rag_products = []

        # 6b) Build context and final prompt
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

        # --- NEW: Save final prompt for reproducibility ---
        PROMPTS_DIR = THIS_DIR / "prompts"
        PROMPTS_DIR.mkdir(exist_ok=True)

        prompt_filename = PROMPTS_DIR / f"{exp_id}.txt"

        try:
            with prompt_filename.open("w", encoding="utf-8") as pf:
                pf.write(final_prompt)
            print(f"   📄 Saved final prompt to: prompts/{prompt_filename.name}")
        except Exception as e:
            print(f"   ⚠️ Failed to save prompt file: {e}")

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

        # Combine any RAG error + Space error into one string for the CSV.
        if rag_error_text:
            if error_text:
                error_text = f"RAG_ERROR: {rag_error_text} | SPACE_ERROR: {error_text}"
            else:
                error_text = f"RAG_ERROR: {rag_error_text}"

        # === NEW (D1): compute quantitative metric using eval.jsonl ===
        ideal_answer = ""
        metric_token_overlap = ""

        if scenario_id and eval_map:
            eval_rec = eval_map.get(scenario_id)
            if eval_rec:
                ideal_answer = eval_rec.get("ideal_answer", "") or ""
                if ideal_answer and response_text:
                    score = token_overlap_f1(ideal_answer, response_text)
                    metric_token_overlap = f"{score:.4f}"
                    print(f"   [Metric] token_overlap_f1 = {metric_token_overlap}")
            else:
                print(f"   ⚠️ No eval record found for scenario_id={scenario_id!r}")

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

        # NEW (D1) columns
        result_row["scenario_id"] = scenario_id
        result_row["ideal_answer"] = ideal_answer
        result_row["metric_token_overlap_f1"] = metric_token_overlap
        # human eval to be filled later
        result_row["helpfulness_score"] = ""  # 1–5 (you will fill manually)
        result_row["factuality_score"] = ""  # 1–5 (you will fill manually)

        writer.writerow(result_row)
        out_file.flush()
        new_runs += 1

        # === NEW (D1): Log to MLflow ===
        if mlflow is not None:
            with mlflow.start_run(run_name=exp_id):
                mlflow.log_params(
                    {
                        "strategy": strategy,
                        "few_shot_k": few_shot_k,
                        "sample_type": sample_type or "",
                        "scenario_id": scenario_id or "",
                    }
                )
                if latency_seconds is not None:
                    mlflow.log_metric("latency_seconds", float(latency_seconds))
                if metric_token_overlap:
                    mlflow.log_metric("metric_token_overlap_f1", float(metric_token_overlap))

        print(f"   ✅ Completed experiment {exp_id} (error={bool(error_text)})")
        print("--------------------------------------------------\n")

    out_file.close()
    print("==============================================")
    print(f"🏁 All done. New experiments run in this session: {new_runs}")
    print(f"📄 Results saved in: {RESULTS_PATH}")
    print("==============================================")


if __name__ == "__main__":
    main()
