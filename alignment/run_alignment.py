"""
Alignment runner: library API and CLI. Populates cache, runs optimizers,
saves splits/manifests/results, and writes the report.
"""

import argparse
import json
import os
import pickle
import sys
import uuid
import joblib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add project root for imports when run as script or from notebook
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alignment.config_space import vector_to_config, config_to_dict, config_to_vector
from alignment.scoring import (
    compute_correlation_for_samples,
    save_to_cache,
    cache_path,
)
from alignment.evaluate import evaluate_best_configs, compute_per_sample_scores, write_report


def _get_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _get_task(task_name: str):
    if task_name == "summeval":
        from alignment.tasks.summeval_task import SummEvalTask
        return SummEvalTask()
    raise ValueError(f"Unknown task: {task_name}")


def _create_llama_config(
    llama_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    model_path: Optional[str] = None,
    num_judges_per_category: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 2000,
):  # -> TrustScoreConfig
    """
    Build TrustScoreConfig with LLaMA-backed judges (VLLM for HuggingFace model ID, or LLAMA for local path).
    Uses DEFAULT_CONFIG as base and replaces span_tagger and judges.
    """
    import copy
    from config.settings import (
        TrustScoreConfig,
        JudgeConfig,
        SpanTaggerConfig,
        LLMProvider,
        DEFAULT_CONFIG,
    )
    if model_path:
        provider = LLMProvider.LLAMA
        model = llama_model or "llama"
    else:
        provider = LLMProvider.VLLM
        model = llama_model
    base = copy.deepcopy(DEFAULT_CONFIG)
    span_tagger_config = SpanTaggerConfig(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        model_path=model_path,
    )
    judges = {}
    for category in ["trust", "bias", "explain"]:
        for i in range(1, num_judges_per_category + 1):
            judge_name = f"{category}_judge_{i}"
            judges[judge_name] = JudgeConfig(
                name=judge_name,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                model_path=model_path,
            )
    base.span_tagger = span_tagger_config
    base.judges = judges
    return base


def _ensure_cache(
    task_name: str,
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    cache_dir: str,
    refresh_cache: bool,
    pipeline_config: Any,
    api_key: Optional[str],
    max_concurrent_samples: int = 1,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pipeline.orchestrator import TrustScorePipeline
    from models.llm_record import LLMRecord
    from config.settings import LLMProvider

    # vLLM's LLM object is not thread-safe — keep sequential for GPU providers
    is_vllm = pipeline_config.span_tagger.provider == LLMProvider.VLLM
    workers = 1 if is_vllm else max(1, max_concurrent_samples)

    pipeline = TrustScorePipeline(config=pipeline_config, api_key=api_key, use_mock=False)
    all_samples = train_samples + val_samples + test_samples
    total = len(all_samples)

    # Thread-safe progress counters
    lock = threading.Lock()
    counts = {"done": 0, "cached": 0, "failed": 0}

    def _log_progress(sid: str, status: str) -> None:
        with lock:
            counts["done"] += 1
            if status == "cached":
                counts["cached"] += 1
            elif status == "failed":
                counts["failed"] += 1
            remaining = total - counts["done"]
            print(
                f"[Cache] {status.upper():8s} {counts['done']:>4}/{total}"
                f" | remaining: {remaining}"
                f" | cached: {counts['cached']}"
                f" | failed: {counts['failed']}"
                f" | id: {sid}"
            )

    def _process_sample(args: tuple) -> None:
        i, sample = args
        sid = sample.get("unique_dataset_id") or sample.get("sample_id", str(i))
        path = cache_path(cache_dir, task_name, sid)
        if not refresh_cache and os.path.exists(path):
            _log_progress(sid, "cached")
            return
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        model = sample.get("model", "unknown")
        try:
            out = pipeline.process(prompt, response, model=model)
            if out.graded_spans is not None:
                llm_record = LLMRecord(
                    task_prompt=out.task_prompt,
                    llm_response=out.llm_response,
                    model_metadata=out.model_metadata,
                )
                save_to_cache(cache_dir, task_name, sid, llm_record, out.graded_spans)
            _log_progress(sid, "done")
        except Exception as e:
            _log_progress(sid, "failed")

    print(f"[Cache] Starting inference: {total} samples, {workers} concurrent worker(s)")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_sample, (i, s)) for i, s in enumerate(all_samples)]
        for future in as_completed(futures):
            future.result()  # re-raise any unexpected exceptions


def run(
    task_name: str = "summeval",
    method: str = "all",
    max_samples: Optional[int] = 100,
    max_evals: int = 50,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    cache_dir: str = "alignment/cache",
    results_dir: str = "alignment/results",
    refresh_cache: bool = False,
    skip_inference: bool = False,
    splits_manifest_path: Optional[str] = None,
    api_key: Optional[str] = None,
    use_llama: bool = False,
    llama_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    model_path: Optional[str] = None,
    num_judges_per_category: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    max_concurrent_samples: int = 1,
    max_concurrent_api_calls: int = 8,
) -> Dict[str, Any]:
    """
    Main entry point. Loads task, splits (or loads from splits_manifest_path), populates cache,
    runs selected optimizers, evaluates best configs on train/val/test, writes artifacts.
    Returns a result dict with run_id, results_dir, paths, best_configs, alignment_results.

    If skip_inference is True, the cache population step is skipped entirely --
    the existing cache_dir must already contain inference results for all samples.

    If use_llama is True, pipeline uses LLaMA (VLLM with HuggingFace model ID, or LLAMA provider
    with model_path) instead of the default OpenAI config.
    """
    from config.settings import load_config
    run_id = _get_run_id()
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    task = _get_task(task_name)
    if splits_manifest_path and os.path.exists(splits_manifest_path):
        with open(splits_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_ids = set(manifest["train_ids"])
        val_ids = set(manifest["val_ids"])
        test_ids = set(manifest["test_ids"])
        # Load enough samples to include all split IDs (use None to load full dataset when reusing splits)
        all_samples = task.load_samples(max_samples=None, random_seed=random_seed)
        train_samples = [s for s in all_samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in train_ids]
        val_samples = [s for s in all_samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in val_ids]
        test_samples = [s for s in all_samples if (s.get("unique_dataset_id") or s.get("sample_id", "")) in test_ids]
        # Copy manifest into run_dir so the run is self-contained and notebook can always load run_dir/splits.json
        splits_path = os.path.join(run_dir, "splits.json")
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    else:
        all_samples = task.load_samples(max_samples=max_samples, random_seed=random_seed)
        train_samples, val_samples, test_samples = task.get_splits(
            all_samples, train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed
        )
        manifest = {
            "train_ids": [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(train_samples)],
            "val_ids": [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(val_samples)],
            "test_ids": [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(test_samples)],
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "random_seed": random_seed,
            "task": task_name,
            "max_samples": max_samples,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
        }
        splits_path = os.path.join(run_dir, "splits.json")
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if use_llama:
        pipeline_config = _create_llama_config(
            llama_model=llama_model,
            model_path=model_path,
            num_judges_per_category=num_judges_per_category,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        pipeline_config = load_config()

    # Build a model-aware cache subdirectory so different judge models /
    # sample counts never overwrite each other.
    # Layout: <cache_dir>/<model_slug>_n<total_samples>/
    if use_llama:
        _model_slug = llama_model.replace("/", "_").replace("\\", "_")
    else:
        _model_slug = pipeline_config.span_tagger.model.replace("/", "_").replace("\\", "_")
    _n_total = len(train_samples) + len(val_samples) + len(test_samples)
    cache_dir = os.path.join(cache_dir, f"{_model_slug}_n{_n_total}")
    print(f"[cache] Using cache directory: {cache_dir}")

    # Run manifest (written after cache_dir is finalized)
    run_manifest = {
        "run_id": run_id,
        "task": task_name,
        "method": method,
        "max_samples": max_samples,
        "max_evals": max_evals,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "random_seed": random_seed,
        "cache_dir": cache_dir,
        "results_dir": results_dir,
        "refresh_cache": refresh_cache,
        "splits_manifest_path": splits_manifest_path,
        "use_llama": use_llama,
        "llama_model": llama_model if use_llama else None,
        "model_path": model_path if use_llama else None,
        "num_judges_per_category": num_judges_per_category if use_llama else None,
        "judge_model": llama_model if use_llama else pipeline_config.span_tagger.model,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import subprocess
        run_manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        run_manifest["git_commit"] = None
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    if skip_inference:
        print(f"[skip-inference] Skipping cache population; reading from {cache_dir}")
    else:
        from config.settings import LLMProvider
        from modules.llm_providers.openai_provider import set_max_concurrent_api_calls
        if not use_llama:
            set_max_concurrent_api_calls(max_concurrent_api_calls)
        _ensure_cache(
            task_name, train_samples, val_samples, test_samples,
            cache_dir, refresh_cache, pipeline_config, api_key,
            max_concurrent_samples=max_concurrent_samples,
        )

    def make_evaluate_fn(samples: List[Dict]):
        def evaluate_fn(config):
            result = compute_correlation_for_samples(
                config, samples, task, cache_dir, task_name, use_quality=True
            )
            return result[0], result[1]  # (pearson, spearman) for optimizer compatibility
        return evaluate_fn

    configs_log_path = os.path.join(run_dir, "configs_evaluated.jsonl")
    best_configs = {}

    methods_to_run = ["a", "b", "c"] if method == "all" else [method.strip().lower()]
    for m in methods_to_run:
        def _on_eval(vec, cdict, pearson, spearman, _m=m):
            with open(configs_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "config_vector": vec,
                    "config_dict": cdict,
                    "train_pearson": pearson,
                    "train_spearman": spearman,
                    "method": _m,
                    "timestamp": datetime.now().isoformat(),
                }, default=str) + "\n")
        on_eval = _on_eval
        artifact = None
        if m == "a":
            from alignment.optimizers.option_a import run_option_a
            best_vec, best_dict, bp, bs, _ = run_option_a(
                make_evaluate_fn(train_samples),
                max_evals=max_evals,
                random_seed=random_seed,
                on_eval=on_eval,
            )
        elif m == "b":
            from alignment.optimizers.option_b import run_option_b
            best_vec, best_dict, bp, bs, artifact = run_option_b(
                make_evaluate_fn(train_samples),
                max_evals=max_evals,
                random_seed=random_seed,
                on_eval=on_eval,
            )
            if artifact is not None:
                pkl_path = os.path.join(run_dir, "lgb_model.pkl")
                joblib.dump(artifact, pkl_path)
                print(f"[artifact] LightGBM model saved to {pkl_path}")
        elif m == "c":
            from alignment.optimizers.option_c import run_option_c
            best_vec, best_dict, bp, bs, artifact = run_option_c(
                make_evaluate_fn(train_samples),
                max_evals=max_evals,
                random_seed=random_seed,
                on_eval=on_eval,
            )
            if artifact is not None:
                pkl_path = os.path.join(run_dir, "optuna_study.pkl")
                with open(pkl_path, "wb") as pf:
                    pickle.dump(artifact, pf)
                print(f"[artifact] Optuna study saved to {pkl_path}")
        else:
            continue
        best_configs[m] = {"config_vector": best_vec, "config_dict": best_dict}

    # Include default (pipeline) config so report shows baseline correlations
    default_vec = config_to_vector(pipeline_config)
    default_dict = config_to_dict(pipeline_config)
    best_configs = {"default": {"config_vector": default_vec, "config_dict": default_dict}, **best_configs}

    with open(os.path.join(run_dir, "best_configs.json"), "w", encoding="utf-8") as f:
        json.dump(best_configs, f, indent=2)

    alignment_results = evaluate_best_configs(
        best_configs, train_samples, val_samples, test_samples,
        task, cache_dir, task_name,
    )
    with open(os.path.join(run_dir, "alignment_results.json"), "w", encoding="utf-8") as f:
        json.dump(alignment_results, f, indent=2)

    # Save per-sample scores for each method (enables downstream analysis)
    all_samples = train_samples + val_samples + test_samples
    sample_scores = {}
    for method_name, data in best_configs.items():
        vec = data.get("config_vector")
        if not vec:
            continue
        config = vector_to_config(vec)
        sample_scores[method_name] = compute_per_sample_scores(
            config, all_samples, task, cache_dir, task_name
        )
    sample_scores_path = os.path.join(run_dir, "sample_scores.json")
    with open(sample_scores_path, "w", encoding="utf-8") as f:
        json.dump(sample_scores, f, indent=2)
    print(f"[artifact] Per-sample scores saved to {sample_scores_path}")

    write_report(
        results_dir, run_id, task_name,
        len(train_samples), len(val_samples), len(test_samples),
        best_configs, alignment_results,
    )

    return {
        "run_id": run_id,
        "results_dir": results_dir,
        "run_dir": run_dir,
        "splits_manifest_path": os.path.join(run_dir, "splits.json"),
        "configs_evaluated_path": configs_log_path,
        "best_configs": best_configs,
        "alignment_results": alignment_results,
        "alignment_report_path": os.path.join(run_dir, "alignment_report.md"),
        "alignment_results_path": os.path.join(run_dir, "alignment_results.json"),
        "sample_scores_path": sample_scores_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run alignment: learn configs that maximize correlation with human scores.")
    parser.add_argument("--task", default="summeval", help="Task name (e.g. summeval)")
    parser.add_argument("--method", default="all", choices=["a", "b", "c", "all"], help="Optimizer: a, b, c, or all")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples (None = all)")
    parser.add_argument("--max-evals", type=int, default=50, help="Max evaluations per optimizer")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="alignment/cache")
    parser.add_argument("--results-dir", default="alignment/results")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-inference", action="store_true", help="Skip cache population; read inference results from existing cache-dir")
    parser.add_argument("--splits-manifest", type=str, default=None, help="Path to existing splits.json to reuse")
    parser.add_argument("--api-key", type=str, default=None, help="API key for pipeline (OpenAI; not needed when using LLaMA)")
    parser.add_argument("--use-llama", action="store_true", help="Use LLaMA (VLLM or local) instead of OpenAI for judges")
    parser.add_argument("--llama-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID for VLLM (when --use-llama without --model-path)")
    parser.add_argument("--model-path", type=str, default=None, help="Local model path for LLaMA provider (optional)")
    parser.add_argument("--num-judges-per-category", type=int, default=3, help="Judges per category when using LLaMA")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for LLM generation (span tagger and judges)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM generation")
    args = parser.parse_args()
    result = run(
        task_name=args.task,
        method=args.method,
        max_samples=args.max_samples,
        max_evals=args.max_evals,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        refresh_cache=args.refresh_cache,
        skip_inference=args.skip_inference,
        splits_manifest_path=args.splits_manifest,
        api_key=args.api_key,
        use_llama=args.use_llama,
        llama_model=args.llama_model,
        model_path=args.model_path,
        num_judges_per_category=args.num_judges_per_category,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("Run ID:", result["run_id"])
    print("Run dir:", result["run_dir"])
    print("Report:", result["alignment_report_path"])


if __name__ == "__main__":
    main()
