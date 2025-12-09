# src/llm/experiments/run_experiments.py
import csv
import datetime
import json
import os
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

from gradio_client import Client

# Import your existing ML, RAG, and merge logic
from ...ml.service import get_ml_candidates_for_user
from ...rag.rag_service import ask
from ..advisor import merge_ml_and_rag

print("ml rag llm imported!")

# ================================
# 0. Optional: MLflow for logging
# ================================
try:
    import mlflow
except ImportError:
    mlflow = None
print("mlflow imported!")

# ================================
# 1. Paths & constants
# ================================

print("creating folders...")

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

# CSV with high-level experiment results & metrics (no prompt/response content)
RESULTS_PATH = THIS_DIR / "experiment_results.csv"

# Few-shot examples for prompts
SAMPLE_RESPONSES_PATH = THIS_DIR / "sample_responses.json"

# Directory to store full prompts (one .txt per experiment_id)
PROMPTS_DIR = THIS_DIR / "prompts"

# Directory to store rich JSON logs (one .json per experiment_id)
EXPERIMENT_LOGS_DIR = THIS_DIR / "experiment_logs"

# Ensure folders exist (safe even if already present)
PROMPTS_DIR.mkdir(exist_ok=True)
EXPERIMENT_LOGS_DIR.mkdir(exist_ok=True)

# Try to load eval.jsonl from data/eval.jsonl (spec requirement),
# but fall back to experiments/ if needed.
PROJECT_ROOT = THIS_DIR.parents[2]
EVAL_PATH_CANDIDATES = [
    PROJECT_ROOT / "data" / "eval.jsonl",
    THIS_DIR / "eval.jsonl",
]
print("folders created!")

print("loading .env...")

# Default user_id if none provided in CSV
DEFAULT_USER_ID = os.getenv(
    "EXPERIMENT_DEFAULT_USER_ID",
    "AG3D6O4STAQKAY2UVGEUV46KN35Q",
)

# Hugging Face Space settings (Haaris' model)
SPACE_ID = os.getenv("QWEN_SPACE_ID", "MuhammadHaaris/mlops")
SPACE_API_NAME = os.getenv("QWEN_API_NAME", "/predict")
print(".env loaded!")

# Single shared client for all calls
# We also configure HTTP-level timeouts via httpx_kwargs so that low-level
# connections don't hang forever.
print("creating client...")
_qwen_client = Client(
    SPACE_ID,
    httpx_kwargs={
        "timeout": 610.0,  # seconds; slightly above our per-attempt job timeout
    },
)
print("client created!")

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

    Behaviour:
    - Uses Client.submit() + Job.result(timeout=...) to enforce a HARD timeout
      per attempt (default 600s = 10 minutes).
    - Retries transient failures up to `max_retries` times with a fixed backoff.
    - Raises the last exception if all attempts fail.
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
            # TimeoutError often has empty str(e), so keep the class name
            print(f"        [call_space] Error: {repr(e)}")

        except Exception as e:
            # Any other network / client / HF error
            elapsed = time.time() - start
            last_error = e
            print(f"        [call_space] ❌ Failed on attempt {attempt} after {elapsed:.3f}s")
            print(f"        [call_space] Error: {repr(e)}")

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
    """Load few-shot examples from sample_responses.json (if present)."""
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
    Select up to k examples matching the given sample_type and
    (optionally) only positive reviews.
    """
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

    # If not enough examples of this type, fall back to any positive examples
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
    """Format selected examples into a block added to the prompt."""
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
    """
    Build the instruction part of the LLM prompt depending on strategy:
    - zero_shot (default)
    - few_shot_3 / few_shot_5
    - cot (chain-of-thought)
    - meta (meta-prompting / persona-style)
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

    # Determine k from strategy if not explicitly set
    k = few_shot_k or (3 if "3" in strategy else 5 if "5" in strategy else 0)

    # Example-driven few-shot prompting
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

    # Chain-of-thought strategy
    if strategy == "cot":
        prompt += (
            "\n\nWhen answering, first think step-by-step about the options using the ML and "
            "RAG information. Explicitly compare the top candidates. Then end with a short "
            "section titled 'Final Recommendation' with 2–4 sentences."
        )

    # Meta-prompt strategy (persona + rules + format)
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
# 4. Config + eval + CSV helpers
# ================================


def load_config_rows() -> List[Dict[str, str]]:
    """Load experiment configuration rows from CSV."""
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


def load_eval_dataset() -> Dict[str, Dict[str, str]]:
    """
    Load eval.jsonl as a mapping: scenario_id -> record.

    This is used to fetch the ideal_answer and any other metadata.
    """
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


# ================================
# 5. Metrics helpers (token F1)
# ================================


def _normalize_text_for_metric(text: str) -> List[str]:
    """Lowercase, strip punctuation, and split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def token_overlap_f1(ref: str, pred: str) -> float:
    """
    Simple token-overlap F1 as a stand-in for ROUGE/BLEU-style metrics.

    This satisfies the 'automated quantitative metric' requirement, and is
    straightforward to explain in EVALUATION.md.
    """
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
# 6. JSON logging helper
# ================================


def save_experiment_json(
    *,
    exp_id: str,
    row: Dict[str, str],
    user_query: str,
    user_id: str,
    strategy: str,
    sample_type: str | None,
    few_shot_k: int,
    scenario_id: str,
    start_iso: str,
    end_iso: str,
    latency_seconds: float | None,
    final_prompt: str,
    response_text: str,
    ideal_answer: str,
    metric_token_overlap: float | None,
    ml_candidates: List[Dict[str, Any]],
    rag_result: Dict[str, Any],
) -> None:
    """
    Save a rich JSON snapshot for this experiment_id.

    This lets you later inspect, per experiment:
    - What config was used
    - What prompt was sent
    - What RAG/ML context was used
    - What the LLM responded
    - What metrics were computed
    """
    record: Dict[str, Any] = {
        "experiment_id": exp_id,
        "config_row": row,
        "llm": {
            "strategy": strategy,
            "sample_type": sample_type,
            "few_shot_k": few_shot_k,
            "scenario_id": scenario_id,
            "user_id": user_id,
            "user_query": user_query,
        },
        "timestamps": {
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
        },
        "prompt": final_prompt,
        "response_text": response_text,
        "metrics": {
            "latency_seconds": latency_seconds,
            "ideal_answer": ideal_answer,
            "metric_token_overlap_f1": metric_token_overlap,
            # Human-in-the-loop slots (to be filled later by you)
            "helpfulness_score": None,
            "factuality_score": None,
        },
        "context": {
            "ml_candidates": ml_candidates,
            "rag_result": rag_result,
        },
    }

    out_path = EXPERIMENT_LOGS_DIR / f"{exp_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    print(f"   📝 Saved detailed JSON log to: experiment_logs/{out_path.name}")


# ================================
# 7. Main experiment runner
# ================================


def main() -> None:
    """
    Entry point for running all LLM experiments defined in experiments_config.csv.

    High-level flow:
    1) Load config rows and few-shot samples.
    2) Load eval.jsonl for quantitative metrics (if available).
    3) Prepare result schema and figure out which experiment_ids are already done.
    4) For each remaining experiment:
       - Build ML + RAG context.
       - Construct the final LLM prompt (strategy-specific).
       - Call the Qwen HF Space with retries and timeouts.
       - Compute metrics using eval.jsonl.
       - ONLY IF the run is successful:
           * Save a prompt .txt file.
           * Save a detailed JSON log.
           * Append a row to experiment_results.csv (meta + metrics only).
           * Log metrics to MLflow.
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

    # 2.5) Load eval dataset
    print("[STEP 2.5] Loading eval.jsonl for quantitative metrics...")
    eval_map = load_eval_dataset()

    # 3) Prepare result fieldnames / schema for the CSV
    print("[STEP 3] Preparing output result schema...")
    extra_fields = [
        "start_time_iso",
        "end_time_iso",
        "latency_seconds",
        "response_chars",
        "response_text",  # we will keep the column but store "" to avoid duplicating content
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

    # Initialize MLflow experiment (if available)
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

        # Choose user query (prefer variant if present)
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

        # Base LLM prompt (config-specific, optional)
        base_llm_prompt = (row.get("llm_prompt") or "").rstrip()
        print(
            "   base llm_prompt set? : ",
            "YES" if base_llm_prompt else "NO (using default)",
        )

        # Build instruction prompt based on strategy
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

        rag_result: Dict[str, Any] = {}
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
            rag_error_text = repr(e)
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

        if rag_error_text:
            print("RAG HAS AN ERROR. LLM WILL NOT RUN AND EXPERIMENT IS SKIPPED. ")
            continue

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

        # Save final prompt as a .txt file for reproducibility
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
        latency_seconds: float | None = None
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
            # Capture repr(e) so even empty str(e) errors are visible
            error_text = repr(e)
            print("        ❌ Space call FAILED with error:")
            print("           ", error_text)

        end_time = datetime.datetime.now()
        end_iso = end_time.isoformat(timespec="seconds")
        print(f"        End time   : {end_iso}")

        # Combine any RAG error + Space error into one string for reporting/logging
        if rag_error_text:
            if error_text:
                error_text = f"RAG_ERROR: {rag_error_text} | SPACE_ERROR: {error_text}"
            else:
                error_text = f"RAG_ERROR: {rag_error_text}"

        # === Metrics: compute token-overlap F1 if we have eval & a response ===
        ideal_answer = ""
        metric_token_overlap_value: float | None = None

        if scenario_id and eval_map:
            eval_rec = eval_map.get(scenario_id)
            if eval_rec:
                ideal_answer = eval_rec.get("ideal_answer", "") or ""
                if ideal_answer and response_text:
                    score = token_overlap_f1(ideal_answer, response_text)
                    metric_token_overlap_value = score
                    print(f"   [Metric] token_overlap_f1 = {score:.4f}")
            else:
                print(f"   ⚠️ No eval record found for scenario_id={scenario_id!r}")

        # Decide if this run counts as "successful" for logging
        run_success = bool(response_text) and not bool(error_text)

        if not run_success:
            print("   ❌ Experiment did NOT complete successfully.")
            print("      → Skipping CSV/MLflow/JSON logging for this run.")
            print(f"      error_text = {error_text!r}")
            print("--------------------------------------------------\n")
            continue

        # 6d) Build result row for CSV (meta + metrics only; no prompt/response content)
        print("   [6d] Writing result row to CSV...")
        result_row: Dict[str, str] = {}

        for key in config_fieldnames:
            result_row[key] = row.get(key, "")

        result_row["start_time_iso"] = start_iso
        result_row["end_time_iso"] = end_iso
        result_row["latency_seconds"] = (
            f"{latency_seconds:.3f}" if latency_seconds is not None else ""
        )
        # We can still log response length as a metric, but omit the text itself
        result_row["response_chars"] = str(len(response_text))
        result_row["response_text"] = ""  # keep column but don't store content
        result_row["error"] = ""  # successful run, so error empty

        # Metric-related columns
        result_row["scenario_id"] = scenario_id
        result_row["ideal_answer"] = ideal_answer
        result_row["metric_token_overlap_f1"] = (
            f"{metric_token_overlap_value:.4f}" if metric_token_overlap_value is not None else ""
        )
        # Human eval columns to be filled manually later
        result_row["helpfulness_score"] = ""
        result_row["factuality_score"] = ""

        writer.writerow(result_row)
        out_file.flush()

        # Save full JSON snapshot with prompt, response, context, and metrics
        save_experiment_json(
            exp_id=exp_id,
            row=row,
            user_query=user_query,
            user_id=user_id,
            strategy=strategy,
            sample_type=sample_type,
            few_shot_k=few_shot_k,
            scenario_id=scenario_id,
            start_iso=start_iso,
            end_iso=end_iso,
            latency_seconds=latency_seconds,
            final_prompt=final_prompt,
            response_text=response_text,
            ideal_answer=ideal_answer,
            metric_token_overlap=metric_token_overlap_value,
            ml_candidates=ml_candidates,
            rag_result=rag_result,
        )

        # Log to MLflow (only for successful runs)
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
                if metric_token_overlap_value is not None:
                    mlflow.log_metric(
                        "metric_token_overlap_f1",
                        float(metric_token_overlap_value),
                    )

        new_runs += 1
        print(f"   ✅ Completed experiment {exp_id} (logged successfully)")
        print("--------------------------------------------------\n")

    out_file.close()
    print("==============================================")
    print(f"🏁 All done. New experiments run in this session: {new_runs}")
    print(f"📄 Results saved in: {RESULTS_PATH}")
    print(f"🗂 JSON logs saved per experiment in: {EXPERIMENT_LOGS_DIR}")
    print("==============================================")


if __name__ == "__main__":
    main()
