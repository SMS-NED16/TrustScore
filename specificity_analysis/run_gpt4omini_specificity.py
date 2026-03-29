"""
GPT-4o-mini Specificity Analysis — Parallelized

Replicates the LLaMA 3.1 8B specificity analysis using GPT-4o-mini via the
OpenAI API, with concurrent inference to keep wall-clock time low.

Usage (from project root):
    python specificity_analysis/run_gpt4omini_specificity.py \
        --num-samples 100 \
        --num-judges 1 \
        --max-workers 20 \
        --output-dir results/specificity_gpt4omini

The script mirrors the paper's protocol exactly:
  - 100 SummEval samples
  - Four perturbed datasets: T, E, B, PLACEBO
  - One error per response injected by GPT-4o-mini
  - TEBScore evaluated with GPT-4o-mini span tagger + judges
  - Score deltas compared to produce specificity matrix (Table 2 analogue)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Ensure project root is on sys.path when called as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import (
    JudgeConfig,
    LLMConfig,
    LLMProvider,
    SpanTaggerConfig,
    TrustScoreConfig,
)
from modules.llm_providers.openai_provider import set_max_concurrent_api_calls
from pipeline.orchestrator import TrustScorePipeline
from specificity_analysis.error_injector import ErrorInjector
from specificity_analysis.load_dataset import load_and_sample_dataset, save_samples
from specificity_analysis.score_comparison import compare_scores, generate_report


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_gpt4omini_config(num_judges: int = 1) -> TrustScoreConfig:
    """Build a TrustScoreConfig that uses GPT-4o-mini for all components."""
    judges: Dict[str, JudgeConfig] = {}
    for i in range(1, num_judges + 1):
        judges[f"trust_judge_{i}"] = JudgeConfig(
            name=f"trust_judge_{i}",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )
        judges[f"bias_judge_{i}"] = JudgeConfig(
            name=f"bias_judge_{i}",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )
        judges[f"explain_judge_{i}"] = JudgeConfig(
            name=f"explain_judge_{i}",
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
        )

    span_tagger = SpanTaggerConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
    )

    return TrustScoreConfig(span_tagger=span_tagger, judges=judges)


# ---------------------------------------------------------------------------
# Parallel inference helpers
# ---------------------------------------------------------------------------

def _process_one_sample(
    args: Tuple[int, Dict[str, Any], TrustScorePipeline, bool]
) -> Dict[str, Any]:
    """Worker: run pipeline.process() on a single sample and return result dict."""
    i, sample, pipeline, is_baseline = args
    unique_id = sample.get(
        "unique_dataset_id",
        f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}",
    )
    try:
        result = pipeline.process(
            prompt=sample["prompt"],
            response=sample["response"],
            model=sample.get("model", "unknown"),
            generated_on=datetime.now(),
        )
        return {
            "idx": i,
            "unique_dataset_id": unique_id,
            "sample_id": sample.get("sample_id", f"sample_{i}"),
            "baseline": is_baseline,
            "error_type_injected": sample.get("error_type_injected"),
            "perturbed": not is_baseline,
            "trust_score": result.summary.trust_score,
            "trust_quality": result.summary.trust_quality,
            "agg_score_T": result.summary.agg_score_T,
            "agg_quality_T": result.summary.agg_quality_T,
            "agg_score_E": result.summary.agg_score_E,
            "agg_quality_E": result.summary.agg_quality_E,
            "agg_score_B": result.summary.agg_score_B,
            "agg_quality_B": result.summary.agg_quality_B,
            "trust_score_ci": {
                "lower": result.summary.trust_score_ci.lower,
                "upper": result.summary.trust_score_ci.upper,
            },
            "trust_quality_ci": {
                "lower": result.summary.trust_quality_ci.lower,
                "upper": result.summary.trust_quality_ci.upper,
            },
            "agg_score_T_ci": {
                "lower": result.summary.agg_score_T_ci.lower,
                "upper": result.summary.agg_score_T_ci.upper,
            },
            "agg_quality_T_ci": {
                "lower": result.summary.agg_quality_T_ci.lower,
                "upper": result.summary.agg_quality_T_ci.upper,
            },
            "agg_score_E_ci": {
                "lower": result.summary.agg_score_E_ci.lower,
                "upper": result.summary.agg_score_E_ci.upper,
            },
            "agg_quality_E_ci": {
                "lower": result.summary.agg_quality_E_ci.lower,
                "upper": result.summary.agg_quality_E_ci.upper,
            },
            "agg_score_B_ci": {
                "lower": result.summary.agg_score_B_ci.lower,
                "upper": result.summary.agg_score_B_ci.upper,
            },
            "agg_quality_B_ci": {
                "lower": result.summary.agg_quality_B_ci.lower,
                "upper": result.summary.agg_quality_B_ci.upper,
            },
            "num_errors": len(result.errors),
            "errors": {
                eid: {
                    "type": e.type.value,
                    "subtype": e.subtype,
                    "severity_score": e.severity_score,
                    "severity_bucket": e.severity_bucket.value,
                    "explanation": e.explanation,
                }
                for eid, e in result.errors.items()
            },
            "spans": {
                sid: {
                    "start": span.start,
                    "end": span.end,
                    "type": span.type.value,
                    "subtype": span.subtype,
                    "explanation": span.explanation,
                    "severity_score": span.get_average_severity_score(),
                    "judge_count": len(span.analysis),
                }
                for sid, span in (
                    result.graded_spans.spans.items()
                    if result.graded_spans
                    else {}
                )
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        return {
            "idx": i,
            "unique_dataset_id": unique_id,
            "sample_id": sample.get("sample_id", f"sample_{i}"),
            "baseline": is_baseline,
            "perturbed": not is_baseline,
            "error_type_injected": sample.get("error_type_injected"),
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }


def run_inference_parallel(
    samples: List[Dict[str, Any]],
    output_path: str,
    config: TrustScoreConfig,
    api_key: Optional[str],
    max_workers: int,
    label: str = "inference",
    is_baseline: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run TrustScore inference on all samples in parallel.

    Resumes automatically if output_path already contains partial results.
    Results are written to disk as they arrive (thread-safe via a Lock).
    """
    # Load existing results to support resuming
    existing_ids: set = set()
    existing_results: List[Dict[str, Any]] = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        existing_results.append(rec)
                        existing_ids.add(rec.get("unique_dataset_id"))
            if existing_ids:
                print(
                    f"  Resuming {label}: {len(existing_ids)} samples already done, "
                    f"{len(samples) - len(existing_ids)} remaining."
                )
        except Exception as exc:
            print(f"  Warning: could not load existing results ({exc}). Starting fresh.")
            existing_results, existing_ids = [], set()

    # Filter out already-processed samples
    todo = [
        s for s in samples
        if s.get(
            "unique_dataset_id",
            f"{s.get('sample_id', 'unknown')}-{s.get('model', 'unknown')}",
        )
        not in existing_ids
    ]

    if not todo:
        print(f"  {label}: all samples already processed. Skipping.")
        return existing_results

    # One pipeline instance per worker thread keeps things simple; alternatively
    # share a single pipeline (OpenAI client is thread-safe).
    pipeline = TrustScorePipeline(config=config, api_key=api_key)

    file_lock = Lock()
    results: List[Dict[str, Any]] = list(existing_results)
    file_mode = "a" if existing_results else "w"

    work = [(i, s, pipeline, is_baseline) for i, s in enumerate(todo)]

    with open(output_path, file_mode, encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_one_sample, args): args for args in work}
            with tqdm(total=len(work), desc=label, unit="sample") as pbar:
                for future in as_completed(futures):
                    rec = future.result()
                    with file_lock:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        results.append(rec)
                    status = "ERR" if "error" in rec else f"T={rec.get('agg_quality_T', 0):.1f}"
                    pbar.set_postfix_str(status)
                    pbar.update(1)

    successes = sum(1 for r in results if "error" not in r)
    print(f"  {label} complete: {successes}/{len(results)} succeeded → {output_path}")
    return results


# ---------------------------------------------------------------------------
# Parallel error injection
# ---------------------------------------------------------------------------

def _inject_one(
    args: Tuple[int, Dict[str, Any], str, ErrorInjector]
) -> Dict[str, Any]:
    """Worker: inject one error into one sample."""
    i, sample, error_type, injector = args
    original = sample["response"]
    try:
        if error_type == "T":
            modified, subtype = injector.inject_trustworthiness_error(original)
        elif error_type == "B":
            modified, subtype = injector.inject_bias_error(original)
        elif error_type == "E":
            modified, subtype = injector.inject_explainability_error(original)
        else:  # PLACEBO
            modified, subtype = injector.inject_placebo(original)

        change = injector._generate_change_description(original, modified, error_type)
        perturbed = sample.copy()
        perturbed["response"] = modified
        perturbed["error_type_injected"] = error_type
        perturbed["error_subtype_injected"] = subtype
        perturbed["original_response"] = original
        perturbed["change_description"] = change
        perturbed.setdefault(
            "unique_dataset_id",
            f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}",
        )
        return {"idx": i, "sample": perturbed}
    except Exception as exc:
        perturbed = sample.copy()
        perturbed["error_type_injected"] = error_type
        perturbed["injection_failed"] = True
        perturbed["change_description"] = f"Injection failed: {exc}"
        perturbed.setdefault(
            "unique_dataset_id",
            f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}",
        )
        return {"idx": i, "sample": perturbed}


def inject_errors_parallel(
    samples: List[Dict[str, Any]],
    error_type: str,
    injector: ErrorInjector,
    output_path: str,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Inject errors into all samples concurrently and save to JSONL."""
    if os.path.exists(output_path):
        print(f"  {error_type}_perturbed already exists — loading from disk.")
        perturbed = []
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                perturbed.append(json.loads(line.strip()))
        return perturbed

    work = [(i, s, error_type, injector) for i, s in enumerate(samples)]
    results_by_idx: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_inject_one, args): args for args in work}
        with tqdm(total=len(work), desc=f"Inject {error_type}", unit="sample") as pbar:
            for future in as_completed(futures):
                rec = future.result()
                results_by_idx[rec["idx"]] = rec["sample"]
                pbar.update(1)

    ordered = [results_by_idx[i] for i in range(len(samples))]

    with open(output_path, "w", encoding="utf-8") as fh:
        for s in ordered:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    failures = sum(1 for s in ordered if s.get("injection_failed"))
    print(f"  {error_type}_perturbed: {len(ordered) - failures}/{len(ordered)} succeeded → {output_path}")
    return ordered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_gpt4omini_specificity(
    num_samples: int = 100,
    num_judges: int = 1,
    max_workers: int = 20,
    output_dir: str = "results/specificity_gpt4omini",
    api_key: Optional[str] = None,
    skip_injection: bool = False,
    skip_inference: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    print("=" * 70)
    print("GPT-4o-mini SPECIFICITY ANALYSIS  (parallelized)")
    print("=" * 70)
    print(f"  samples={num_samples}  judges/dim={num_judges}  workers={max_workers}")
    print(f"  output → {output_dir}")
    print()

    config = make_gpt4omini_config(num_judges=num_judges)

    # Align the global OpenAI API semaphore with the requested parallelism.
    # Each pipeline.process() call makes (1 span-tagger + num_judges×3) API
    # calls internally; the semaphore caps simultaneous in-flight HTTP requests
    # across all threads.  Using max_workers here is a reasonable upper bound;
    # lower it to 8–10 if you are on OpenAI Tier 1 (500 RPM).
    set_max_concurrent_api_calls(max_workers)

    # ------------------------------------------------------------------
    # Step 0: Load dataset
    # ------------------------------------------------------------------
    samples_path = os.path.join(output_dir, "sampled_dataset.jsonl")
    if os.path.exists(samples_path):
        print("[Step 0] Loading existing sampled dataset...")
        samples = []
        with open(samples_path, "r", encoding="utf-8") as fh:
            for line in fh:
                samples.append(json.loads(line.strip()))
        print(f"  Loaded {len(samples)} samples from disk.")
    else:
        print("[Step 0] Loading and sampling dataset...")
        samples = load_and_sample_dataset(
            dataset_name="summeval",
            max_samples=num_samples,
        )
        save_samples(samples, samples_path)
        print(f"  Sampled {len(samples)} examples → {samples_path}")

    # ------------------------------------------------------------------
    # Step 1: Baseline inference (parallel)
    # ------------------------------------------------------------------
    baseline_path = os.path.join(output_dir, "baseline_results.jsonl")
    if not skip_inference:
        print("\n[Step 1] Baseline inference (GPT-4o-mini, parallel)...")
        run_inference_parallel(
            samples=samples,
            output_path=baseline_path,
            config=config,
            api_key=api_key,
            max_workers=max_workers,
            label="Baseline",
            is_baseline=True,
        )
    else:
        print("\n[Step 1] Skipping baseline inference (--skip-inference).")

    # ------------------------------------------------------------------
    # Step 2: Error injection (parallel, all 4 types simultaneously)
    # ------------------------------------------------------------------
    error_types = ["T", "B", "E", "PLACEBO"]
    perturbed_paths: Dict[str, str] = {
        et: os.path.join(output_dir, f"{et}_perturbed.jsonl") for et in error_types
    }

    if not skip_injection:
        print("\n[Step 2] Error injection (GPT-4o-mini, parallel)...")
        injector = ErrorInjector(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key=api_key,
        )
        perturbed_datasets: Dict[str, List[Dict[str, Any]]] = {}
        for et in error_types:
            perturbed_datasets[et] = inject_errors_parallel(
                samples=samples,
                error_type=et,
                injector=injector,
                output_path=perturbed_paths[et],
                max_workers=max_workers,
            )
    else:
        print("\n[Step 2] Skipping injection (--skip-injection). Loading from disk...")
        perturbed_datasets = {}
        for et in error_types:
            perturbed = []
            with open(perturbed_paths[et], "r", encoding="utf-8") as fh:
                for line in fh:
                    perturbed.append(json.loads(line.strip()))
            perturbed_datasets[et] = perturbed

    # ------------------------------------------------------------------
    # Step 3: Perturbed inference (parallel, each type)
    # ------------------------------------------------------------------
    perturbed_results_paths: Dict[str, str] = {
        et: os.path.join(output_dir, f"{et}_perturbed_results.jsonl")
        for et in error_types
    }

    if not skip_inference:
        print("\n[Step 3] Perturbed inference (GPT-4o-mini, parallel)...")
        for et in error_types:
            run_inference_parallel(
                samples=perturbed_datasets[et],
                output_path=perturbed_results_paths[et],
                config=config,
                api_key=api_key,
                max_workers=max_workers,
                label=f"{et}_perturbed",
                is_baseline=False,
            )
    else:
        print("\n[Step 3] Skipping perturbed inference (--skip-inference).")

    # ------------------------------------------------------------------
    # Step 4: Score comparison
    # ------------------------------------------------------------------
    print("\n[Step 4] Comparing scores...")
    comparisons: Dict[str, Any] = {}
    for et in error_types:
        try:
            comparisons[et] = compare_scores(
                baseline_results_path=baseline_path,
                perturbed_results_path=perturbed_results_paths[et],
                error_type=et,
                config=config,
            )
        except Exception as exc:
            print(f"  Error comparing {et}: {exc}")
            comparisons[et] = {"error": str(exc)}

    report_path = os.path.join(output_dir, "specificity_report.json")
    generate_report(comparisons, report_path)

    # ------------------------------------------------------------------
    # Step 5: Print specificity matrix (Table 2 analogue)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPECIFICITY MATRIX (mean quality-score decrease, pp)")
    print("Rows = error injected | Columns = dimension measured")
    print("-" * 70)
    print(f"{'Injected':10s} | {'ΔT':>8s} | {'ΔE':>8s} | {'ΔB':>8s} | {'Specific?':>10s}")
    print("-" * 70)
    for et in ["T", "E", "B", "PLACEBO"]:
        comp = comparisons.get(et, {})
        spec = comp.get("specificity_analysis", {})
        dT = spec.get(f"T_quality_decrease_when_{et}_injected", float("nan"))
        dE = spec.get(f"E_quality_decrease_when_{et}_injected", float("nan"))
        dB = spec.get(f"B_quality_decrease_when_{et}_injected", float("nan"))
        if et == "PLACEBO":
            is_s = "✓" if spec.get("is_placebo_effective", False) else "✗"
        else:
            is_s = "✓" if spec.get("is_specific", False) else "✗"
        print(f"{et:10s} | {dT:+8.2f} | {dE:+8.2f} | {dB:+8.2f} | {is_s:>10s}")
    print("=" * 70)
    print(f"Full report → {report_path}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini parallelized specificity analysis"
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--num-judges",
        type=int,
        default=1,
        help="Judges per dimension (1 = lean/fast, 3 = matches paper)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Concurrent API calls (throttle if hitting rate limits)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/specificity_gpt4omini",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--skip-injection",
        action="store_true",
        help="Skip error injection (reuse existing perturbed datasets)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip all inference steps (only re-run score comparison)",
    )
    args = parser.parse_args()

    run_gpt4omini_specificity(
        num_samples=args.num_samples,
        num_judges=args.num_judges,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        skip_injection=args.skip_injection,
        skip_inference=args.skip_inference,
    )


if __name__ == "__main__":
    main()
