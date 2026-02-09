"""
Computational Profiling Script for TrustScore Pipeline

This script profiles the TrustScore pipeline to collect:
- Latency metrics (end-to-end and per-stage)
- LLM call counts
- Token usage (input/output)
- Component-level breakdowns

Designed for Jupyter notebook execution on RunPod.
Can be run as: exec(open('computational_profile.py').read())
OR copy-paste cells marked with "# Cell N:" into notebook.
"""

# Cell 1: Imports and setup
import time
import json
import platform
import subprocess
import sys
import os
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import random

# TrustScore imports
from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig, JudgeConfig, LLMProvider
from models.llm_record import LLMRecord, SpansLevelTags, GradedSpans

# Try to import tiktoken for token counting (fallback if not available)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[WARNING] tiktoken not available, will use character-based approximation for token counting")


# Cell 2: Profiler class definition
class PipelineProfiler:
    """Profiles TrustScore pipeline execution without modifying underlying code"""
    
    def __init__(self, config: TrustScoreConfig, api_key: Optional[str] = None, use_mock: bool = False):
        self.config = config
        self.api_key = api_key
        self.use_mock = use_mock
        
        # Initialize pipeline
        self.pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
        
        # Metrics storage
        self.response_metrics: List[Dict[str, Any]] = []
        self.stage_metrics: Dict[str, List[float]] = defaultdict(list)
        self.llm_call_counts: Dict[str, int] = defaultdict(int)
        self.token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        
        # Track LLM calls and tokens by intercepting provider methods
        self._setup_profiling_hooks()
        
    def _setup_profiling_hooks(self):
        """Set up hooks to track LLM calls and tokens without modifying pipeline code"""
        # We'll track calls and tokens per response by intercepting at the provider level
        # This is done by wrapping provider instances after pipeline initialization
        pass  # Token/call tracking will be done per-response in profile_response
    
    def _count_tokens_approximate(self, text: str, model: str = "gpt-4o") -> int:
        """Approximate token count using tiktoken or character-based estimate"""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except:
                pass
        # Fallback: rough approximation (1 token ≈ 4 characters for English)
        return len(text) // 4
    
    def _instrument_stage(self, stage_name: str, func, *args, **kwargs):
        """Instrument a pipeline stage to measure latency and collect metrics"""
        start_time = time.monotonic()
        
        # Set stage context for LLM call tracking
        if hasattr(self.pipeline, 'span_tagger') and hasattr(self.pipeline.span_tagger, 'llm_provider'):
            self.pipeline.span_tagger.llm_provider._current_stage = stage_name
        for judge_type in ['trustworthiness', 'bias', 'explainability']:
            for judge in self.pipeline.judges.get(judge_type, {}).values():
                if hasattr(judge, 'llm_provider'):
                    judge.llm_provider._current_stage = stage_name
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Clear stage context
            if hasattr(self.pipeline, 'span_tagger') and hasattr(self.pipeline.span_tagger, 'llm_provider'):
                self.pipeline.span_tagger.llm_provider._current_stage = None
            for judge_type in ['trustworthiness', 'bias', 'explainability']:
                for judge in self.pipeline.judges.get(judge_type, {}).values():
                    if hasattr(judge, 'llm_provider'):
                        judge.llm_provider._current_stage = None
        
        end_time = time.monotonic()
        duration = end_time - start_time
        self.stage_metrics[stage_name].append(duration)
        
        return result, duration
    
    def _wrap_provider_for_profiling(self, provider, stage_name: str):
        """Wrap an LLM provider to track calls and tokens for a specific stage"""
        if not hasattr(provider, 'generate'):
            return
        
        # Store original if not already stored
        if not hasattr(provider, '_original_generate'):
            provider._original_generate = provider.generate
        
        # For OpenAI provider, also wrap the client's chat.completions.create
        if hasattr(provider, 'client') and hasattr(provider.client, 'chat'):
            if not hasattr(provider.client.chat.completions, '_original_create'):
                provider.client.chat.completions._original_create = provider.client.chat.completions.create
                
                def profiled_create(*args, **kwargs):
                    # Call original
                    response_obj = provider.client.chat.completions._original_create(*args, **kwargs)
                    
                    # Extract token counts
                    if hasattr(response_obj, 'usage') and response_obj.usage:
                        tokens_in = response_obj.usage.prompt_tokens
                        tokens_out = response_obj.usage.completion_tokens
                        self.token_counts[stage_name]["in"] += tokens_in
                        self.token_counts[stage_name]["out"] += tokens_out
                    
                    return response_obj
                
                provider.client.chat.completions.create = profiled_create
        
        # Create wrapper that captures self
        profiler_self = self
        
        def profiled_generate(messages, **kwargs):
            # Track call
            profiler_self.llm_call_counts[stage_name] += 1
            
            # Call original
            result = provider._original_generate(messages, **kwargs)
            
            # If we didn't get tokens from OpenAI response (non-OpenAI provider), approximate
            if not (hasattr(provider, 'client') and hasattr(provider.client, 'chat')):
                full_text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
                approx_tokens_in = profiler_self._count_tokens_approximate(full_text)
                approx_tokens_out = profiler_self._count_tokens_approximate(str(result))
                profiler_self.token_counts[stage_name]["in"] += approx_tokens_in
                profiler_self.token_counts[stage_name]["out"] += approx_tokens_out
            
            return result
        
        provider.generate = profiled_generate
    
    def profile_response(self, prompt: str, response: str, model: str = "unknown", 
                        generation_seed: Optional[int] = None) -> Dict[str, Any]:
        """Profile a single response through the pipeline"""
        
        # Reset per-response counters
        response_start_time = time.monotonic()
        
        # Wrap providers for this response to track calls and tokens
        if hasattr(self.pipeline.span_tagger, 'llm_provider'):
            self._wrap_provider_for_profiling(self.pipeline.span_tagger.llm_provider, "span_tagging")
        
        # Wrap judge providers
        for category in ["trustworthiness", "bias", "explainability"]:
            for judge in self.pipeline.judges.get(category, {}).values():
                if hasattr(judge, 'llm_provider'):
                    self._wrap_provider_for_profiling(judge.llm_provider, "severity_scoring")
        
        # Reset counters for this response
        self.llm_call_counts.clear()
        self.token_counts.clear()
        
        # Stage 1: Span Tagging (includes error classification)
        llm_record = self.pipeline._create_llm_record(prompt, response, model, None)
        spans_tags, span_tagging_time = self._instrument_stage(
            "span_tagging",
            self.pipeline.span_tagger.tag_spans,
            llm_record
        )
        
        # Count spans by type
        span_counts = {"T": 0, "B": 0, "E": 0, "total": len(spans_tags.spans)}
        for span_id, span in spans_tags.spans.items():
            span_counts[span.type.value] = span_counts.get(span.type.value, 0) + 1
        
        # Stage 2: Severity Scoring (judge evaluations)
        graded_spans, severity_scoring_time = self._instrument_stage(
            "severity_scoring",
            self.pipeline._grade_spans,
            llm_record,
            spans_tags,
            generation_seed
        )
        
        # Stage 3: Aggregation
        aggregated_output, aggregation_time = self._instrument_stage(
            "aggregation_and_ci",
            self.pipeline.aggregator.aggregate,
            llm_record,
            graded_spans
        )
        
        response_end_time = time.monotonic()
        total_time = response_end_time - response_start_time
        
        # Collect LLM call and token counts for this response
        stage_calls = {
            "span_tagging": self.llm_call_counts.get("span_tagging", 0),
            "severity_scoring": self.llm_call_counts.get("severity_scoring", 0),
            "aggregation_and_ci": self.llm_call_counts.get("aggregation_and_ci", 0)
        }
        
        stage_tokens = {
            "span_tagging": self.token_counts.get("span_tagging", {"in": 0, "out": 0}).copy(),
            "severity_scoring": self.token_counts.get("severity_scoring", {"in": 0, "out": 0}).copy(),
            "aggregation_and_ci": self.token_counts.get("aggregation_and_ci", {"in": 0, "out": 0}).copy()
        }
        
        # Calculate totals
        total_calls = sum(stage_calls.values())
        total_tokens_in = sum(st["in"] for st in stage_tokens.values())
        total_tokens_out = sum(st["out"] for st in stage_tokens.values())
        
        # If token counting failed, use approximation
        if total_tokens_in == 0 and total_tokens_out == 0:
            # Approximate from prompt/response lengths
            total_tokens_in = self._count_tokens_approximate(prompt + response)
            total_tokens_out = self._count_tokens_approximate(str(aggregated_output.summary))
        
        metrics = {
            "wall_clock_seconds_total": total_time,
            "llm_calls_total": total_calls,
            "tokens_in_total": total_tokens_in,
            "tokens_out_total": total_tokens_out,
            "num_spans": span_counts["total"],
            "num_trust_spans": span_counts["T"],
            "num_bias_spans": span_counts["B"],
            "num_explainability_spans": span_counts["E"],
            "num_error_spans": span_counts["total"],
            "stages": {
                "span_tagging": {
                    "wall_clock_seconds": span_tagging_time,
                    "llm_calls": stage_calls["span_tagging"],
                    "tokens_in": stage_tokens["span_tagging"]["in"],
                    "tokens_out": stage_tokens["span_tagging"]["out"]
                },
                "severity_scoring": {
                    "wall_clock_seconds": severity_scoring_time,
                    "llm_calls": stage_calls["severity_scoring"],
                    "tokens_in": stage_tokens["severity_scoring"]["in"],
                    "tokens_out": stage_tokens["severity_scoring"]["out"]
                },
                "aggregation_and_ci": {
                    "wall_clock_seconds": aggregation_time,
                    "llm_calls": stage_calls["aggregation_and_ci"],
                    "tokens_in": stage_tokens["aggregation_and_ci"]["in"],
                    "tokens_out": stage_tokens["aggregation_and_ci"]["out"]
                }
            }
        }
        
        self.response_metrics.append(metrics)
        return metrics
    
    def profile_batch(self, samples: List[Dict[str, Any]], generation_seed: int = 42) -> List[Dict[str, Any]]:
        """Profile a batch of samples"""
        random.seed(generation_seed)
        
        results = []
        for i, sample in enumerate(samples, 1):
            print(f"\n[Profiling] Sample {i}/{len(samples)}")
            try:
                metrics = self.profile_response(
                    prompt=sample.get("prompt", ""),
                    response=sample.get("response", ""),
                    model=sample.get("model", "unknown"),
                    generation_seed=generation_seed + i if generation_seed else None
                )
                results.append(metrics)
            except Exception as e:
                print(f"[ERROR] Failed to profile sample {i}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"error": str(e)})
        
        return results
    
    def _aggregate_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics (p50, p90, mean, etc.)"""
        if not self.response_metrics:
            return {}
        
        # Filter out errors
        valid_metrics = [m for m in self.response_metrics if "error" not in m]
        if not valid_metrics:
            return {}
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        def compute_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p10": 0.0}
            return {
                "mean": statistics.mean(values),
                "p50": percentile(values, 0.5),
                "p90": percentile(values, 0.9),
                "p10": percentile(values, 0.1)
            }
        
        # End-to-end metrics
        total_times = [m["wall_clock_seconds_total"] for m in valid_metrics]
        total_calls = [m["llm_calls_total"] for m in valid_metrics]
        total_tokens_in = [m["tokens_in_total"] for m in valid_metrics]
        total_tokens_out = [m["tokens_out_total"] for m in valid_metrics]
        
        # Structural drivers
        num_spans = [m["num_spans"] for m in valid_metrics]
        num_trust = [m["num_trust_spans"] for m in valid_metrics]
        num_bias = [m["num_bias_spans"] for m in valid_metrics]
        num_explain = [m["num_explainability_spans"] for m in valid_metrics]
        num_error = [m["num_error_spans"] for m in valid_metrics]
        
        # Stage-level metrics
        stage_times = {
            "span_tagging": [m["stages"]["span_tagging"]["wall_clock_seconds"] for m in valid_metrics],
            "severity_scoring": [m["stages"]["severity_scoring"]["wall_clock_seconds"] for m in valid_metrics],
            "aggregation_and_ci": [m["stages"]["aggregation_and_ci"]["wall_clock_seconds"] for m in valid_metrics]
        }
        
        stage_calls = {
            "span_tagging": [m["stages"]["span_tagging"]["llm_calls"] for m in valid_metrics],
            "severity_scoring": [m["stages"]["severity_scoring"]["llm_calls"] for m in valid_metrics],
            "aggregation_and_ci": [m["stages"]["aggregation_and_ci"]["llm_calls"] for m in valid_metrics]
        }
        
        return {
            "end_to_end": {
                "latency": compute_stats(total_times),
                "calls": compute_stats(total_calls),
                "tokens_in": compute_stats(total_tokens_in),
                "tokens_out": compute_stats(total_tokens_out)
            },
            "structural_drivers": {
                "num_spans": compute_stats(num_spans),
                "num_trust_spans": compute_stats(num_trust),
                "num_bias_spans": compute_stats(num_bias),
                "num_explainability_spans": compute_stats(num_explain),
                "num_error_spans": compute_stats(num_error)
            },
            "stages": {
                "times": {stage: compute_stats(times) for stage, times in stage_times.items()},
                "calls": {stage: compute_stats(calls) for stage, calls in stage_calls.items()}
            }
        }
    
    def _collect_environment_metadata(self) -> Dict[str, Any]:
        """Collect comprehensive environment and configuration metadata"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "hardware": {},
            "software": {},
            "configuration": {},
            "run": {}
        }
        
        # Hardware
        try:
            metadata["hardware"]["cpu"] = {
                "model": platform.processor(),
                "cores": os.cpu_count(),
                "architecture": platform.machine()
            }
        except:
            pass
        
        # GPU info
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split("\n")[0].split(", ")
                metadata["hardware"]["gpu"] = {
                    "name": gpu_info[0] if len(gpu_info) > 0 else "unknown",
                    "memory": gpu_info[1] if len(gpu_info) > 1 else "unknown"
                }
            else:
                metadata["hardware"]["gpu"] = {"available": False}
        except:
            metadata["hardware"]["gpu"] = {"available": False}
        
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                metadata["hardware"]["gpu"]["cuda_driver"] = result.stdout.strip()
        except:
            pass
        
        metadata["hardware"]["system"] = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0]
        }
        
        # Software
        metadata["software"]["python"] = {
            "version": sys.version.split()[0],
            "executable": sys.executable
        }
        
        # Package versions
        package_versions = {}
        for pkg in ["torch", "transformers", "openai", "tiktoken"]:
            try:
                mod = __import__(pkg)
                package_versions[pkg] = getattr(mod, "__version__", "unknown")
            except:
                pass
        metadata["software"]["packages"] = package_versions
        
        # Count actual judges per category from pipeline
        t_count = len(self.pipeline.judges.get("trustworthiness", {}))
        b_count = len(self.pipeline.judges.get("bias", {}))
        e_count = len(self.pipeline.judges.get("explainability", {}))
        metadata["configuration"]["judge_counts"] = {
            "trustworthiness": t_count,
            "bias": b_count,
            "explainability": e_count
        }
        
        # Model names and hyperparameters
        if self.pipeline.span_tagger and hasattr(self.pipeline.span_tagger, 'config'):
            metadata["configuration"]["span_tagger"] = {
                "model": self.pipeline.span_tagger.config.model,
                "provider": str(self.pipeline.span_tagger.config.provider.value) if hasattr(self.pipeline.span_tagger.config.provider, 'value') else str(self.pipeline.span_tagger.config.provider),
                "temperature": self.pipeline.span_tagger.config.temperature,
                "max_tokens": self.pipeline.span_tagger.config.max_tokens,
                "batch_size": self.pipeline.span_tagger.config.batch_size
            }
        
        judge_models = {}
        judge_hyperparams = {}
        for category in ["trustworthiness", "bias", "explainability"]:
            judges = list(self.pipeline.judges.get(category, {}).values())
            if judges:
                models = [judge.config.model for judge in judges]
                judge_models[category] = list(set(models))  # Unique models
                
                # Collect hyperparameters (use first judge's config as representative)
                first_judge = judges[0]
                judge_hyperparams[category] = {
                    "temperature": first_judge.config.temperature,
                    "max_tokens": first_judge.config.max_tokens,
                    "provider": str(first_judge.config.provider.value) if hasattr(first_judge.config.provider, 'value') else str(first_judge.config.provider)
                }
        
        metadata["configuration"]["judge_models"] = judge_models
        metadata["configuration"]["judge_hyperparameters"] = judge_hyperparams
        metadata["configuration"]["aggregation_weights"] = {
            "trustworthiness": self.config.aggregation_weights.trustworthiness,
            "explainability": self.config.aggregation_weights.explainability,
            "bias": self.config.aggregation_weights.bias
        }
        metadata["configuration"]["performance"] = {
            "enable_parallel_processing": self.config.performance.enable_parallel_processing,
            "max_concurrent_judges": self.config.performance.max_concurrent_judges,
            "batch_processing": self.config.performance.batch_processing
        }
        
        # Run metadata
        metadata["run"]["num_samples"] = len(self.response_metrics)
        metadata["run"]["use_mock"] = self.use_mock
        metadata["run"]["token_counting_method"] = "API" if any(self.token_counts.values()) else "approximation"
        
        return metadata
    
    def _generate_outputs(self, config_name: str = "default", output_dir: str = "paper_materials"):
        """Generate JSON and markdown outputs"""
        os.makedirs(output_dir, exist_ok=True)
        
        stats = self._aggregate_statistics()
        metadata = self._collect_environment_metadata()
        
        # JSON output
        json_output = {
            "config": config_name,
            "statistics": stats,
            "metadata": metadata,
            "raw_metrics": self.response_metrics
        }
        
        json_path = os.path.join(output_dir, "runtime_summary.json")
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"\n[Output] Saved JSON to {json_path}")
        
        # Markdown output
        md_path = os.path.join(output_dir, "runtime_summary.md")
        with open(md_path, 'w') as f:
            self._write_markdown_report(f, stats, metadata, config_name)
        print(f"[Output] Saved Markdown to {md_path}")
    
    def _write_markdown_report(self, f, stats: Dict[str, Any], metadata: Dict[str, Any], config_name: str):
        """Write paper-ready markdown report"""
        f.write("# TrustScore Computational Profile\n\n")
        f.write(f"**Configuration**: {config_name}\n")
        f.write(f"**Generated**: {metadata['timestamp']}\n\n")
        
        # Summary Table
        f.write("## Summary Statistics\n\n")
        f.write("| Config | N | p50 latency (s) | p90 latency (s) | p50 calls | p90 calls | p50 tokens in/out | p90 tokens in/out |\n")
        f.write("|--------|---|-----------------|-----------------|-----------|-----------|-------------------|-------------------|\n")
        
        if stats and "end_to_end" in stats:
            e2e = stats["end_to_end"]
            f.write(f"| {config_name} | {metadata['run']['num_samples']} | "
                   f"{e2e['latency']['p50']:.2f} | {e2e['latency']['p90']:.2f} | "
                   f"{e2e['calls']['p50']:.0f} | {e2e['calls']['p90']:.0f} | "
                   f"{e2e['tokens_in']['p50']:.0f}/{e2e['tokens_out']['p50']:.0f} | "
                   f"{e2e['tokens_in']['p90']:.0f}/{e2e['tokens_out']['p90']:.0f} |\n")
        f.write("\n")
        
        # Drivers Table
        f.write("## Structural Drivers\n\n")
        f.write("| Config | mean spans (S) | p10–p90 S | mean T spans | mean B spans | mean E spans | mean total error spans | p10–p90 error spans |\n")
        f.write("|--------|---------------|-----------|--------------|-------------|--------------|----------------------|---------------------|\n")
        
        if stats and "structural_drivers" in stats:
            drivers = stats["structural_drivers"]
            f.write(f"| {config_name} | "
                   f"{drivers['num_spans']['mean']:.1f} | "
                   f"{drivers['num_spans']['p10']:.0f}–{drivers['num_spans']['p90']:.0f} | "
                   f"{drivers['num_trust_spans']['mean']:.1f} | "
                   f"{drivers['num_bias_spans']['mean']:.1f} | "
                   f"{drivers['num_explainability_spans']['mean']:.1f} | "
                   f"{drivers['num_error_spans']['mean']:.1f} | "
                   f"{drivers['num_error_spans']['p10']:.0f}–{drivers['num_error_spans']['p90']:.0f} |\n")
        f.write("\n")
        
        # Component Breakdown
        f.write("## Component Breakdown\n\n")
        f.write("| Component | % time | % calls |\n")
        f.write("|-----------|--------|---------|\n")
        
        if stats and "stages" in stats and "end_to_end" in stats:
            total_time = stats["end_to_end"]["latency"]["mean"]
            total_calls = stats["end_to_end"]["calls"]["mean"]
            
            for stage in ["span_tagging", "severity_scoring", "aggregation_and_ci"]:
                stage_time = stats["stages"]["times"][stage]["mean"]
                stage_calls = stats["stages"]["calls"][stage]["mean"]
                pct_time = (stage_time / total_time * 100) if total_time > 0 else 0
                pct_calls = (stage_calls / total_calls * 100) if total_calls > 0 else 0
                f.write(f"| {stage} | {pct_time:.1f}% | {pct_calls:.1f}% |\n")
        f.write("\n")
        
        # Worked Example
        f.write("## Worked Example\n\n")
        if self.response_metrics:
            # Find median-length response
            valid_metrics = [m for m in self.response_metrics if "error" not in m]
            if valid_metrics:
                sorted_by_length = sorted(valid_metrics, key=lambda x: x.get("num_spans", 0))
                median_idx = len(sorted_by_length) // 2
                example = sorted_by_length[median_idx]
                
                num_judges = sum(metadata["configuration"]["judge_counts"].values())
                f.write(f"> For a response segmented into {example['num_spans']} spans "
                       f"({example['num_trust_spans']} T, {example['num_bias_spans']} B, "
                       f"{example['num_explainability_spans']} E) and evaluated with {num_judges} judges, "
                       f"TrustScore required {example['llm_calls_total']} LLM calls, "
                       f"consumed ~{example['tokens_in_total']}/{example['tokens_out_total']} tokens (in/out), "
                       f"and completed in {example['wall_clock_seconds_total']:.2f} seconds.\n\n")
        
        # Environment and Configuration
        f.write("## Environment and Configuration\n\n")
        f.write("### Hardware\n")
        if "gpu" in metadata["hardware"]:
            gpu = metadata["hardware"]["gpu"]
            if gpu.get("available", True):
                f.write(f"- **GPU**: {gpu.get('name', 'unknown')} ({gpu.get('memory', 'unknown')})\n")
                if "cuda_driver" in gpu:
                    f.write(f"- **CUDA Driver**: {gpu['cuda_driver']}\n")
            else:
                f.write("- **GPU**: Not available\n")
        
        if "cpu" in metadata["hardware"]:
            cpu = metadata["hardware"]["cpu"]
            f.write(f"- **CPU**: {cpu.get('model', 'unknown')} ({cpu.get('cores', 'unknown')} cores)\n")
        
        if "system" in metadata["hardware"]:
            sys_info = metadata["hardware"]["system"]
            f.write(f"- **OS**: {sys_info.get('os', 'unknown')} {sys_info.get('os_version', '')}\n")
        
        f.write("\n### Software\n")
        if "python" in metadata["software"]:
            f.write(f"- **Python**: {metadata['software']['python']['version']}\n")
        if "packages" in metadata["software"]:
            f.write("- **Key Packages**:\n")
            for pkg, version in metadata["software"]["packages"].items():
                f.write(f"  - {pkg}: {version}\n")
        
        f.write("\n### Configuration\n")
        if "span_tagger" in metadata["configuration"]:
            st = metadata["configuration"]["span_tagger"]
            f.write(f"- **Span Tagger**:\n")
            f.write(f"  - Model: {st.get('model', 'unknown')}\n")
            f.write(f"  - Provider: {st.get('provider', 'unknown')}\n")
            f.write(f"  - Temperature: {st.get('temperature', 'N/A')}\n")
            f.write(f"  - Max Tokens: {st.get('max_tokens', 'N/A')}\n")
        
        if "judge_counts" in metadata["configuration"]:
            counts = metadata["configuration"]["judge_counts"]
            f.write(f"- **Judges**: T={counts.get('trustworthiness', 0)}, "
                   f"B={counts.get('bias', 0)}, E={counts.get('explainability', 0)}\n")
        if "judge_models" in metadata["configuration"]:
            f.write("- **Judge Models**:\n")
            for cat, models in metadata["configuration"]["judge_models"].items():
                f.write(f"  - {cat}: {', '.join(models)}\n")
        if "judge_hyperparameters" in metadata["configuration"]:
            f.write("- **Judge Hyperparameters**:\n")
            for cat, hyperparams in metadata["configuration"]["judge_hyperparameters"].items():
                f.write(f"  - {cat}: temp={hyperparams.get('temperature', 'N/A')}, "
                       f"max_tokens={hyperparams.get('max_tokens', 'N/A')}, "
                       f"provider={hyperparams.get('provider', 'N/A')}\n")
        if "aggregation_weights" in metadata["configuration"]:
            weights = metadata["configuration"]["aggregation_weights"]
            f.write(f"- **Aggregation Weights**: T={weights['trustworthiness']:.2f}, "
                   f"E={weights['explainability']:.2f}, B={weights['bias']:.2f}\n")
        if "performance" in metadata["configuration"]:
            perf = metadata["configuration"]["performance"]
            f.write(f"- **Parallel Processing**: {perf.get('enable_parallel_processing', False)}\n")
            f.write(f"- **Max Concurrent Judges**: {perf.get('max_concurrent_judges', 'N/A')}\n")
        
        f.write("\n### Run Metadata\n")
        f.write(f"- **Samples Profiled**: {metadata['run']['num_samples']}\n")
        f.write(f"- **Token Counting Method**: {metadata['run']['token_counting_method']}\n")
        f.write(f"- **Use Mock**: {metadata['run']['use_mock']}\n")


# Cell 3: Configuration and sample data loading
def create_config_with_judge_count(num_judges_per_category: int, 
                                   model: str = "meta-llama/Llama-3.1-8B-Instruct",  # Default to LLaMA
                                   provider: LLMProvider = LLMProvider.LLAMA,  # Default to LLaMA
                                   temperature: float = 0.1,
                                   max_tokens: int = 2000) -> TrustScoreConfig:
    """Create a TrustScoreConfig with specified number of judges per category"""
    from config.settings import TrustScoreConfig, JudgeConfig, SpanTaggerConfig
    
    # Create span tagger config with hyperparameters
    span_tagger_config = SpanTaggerConfig(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create judge configs with hyperparameters
    judges = {}
    for category in ["trust", "bias", "explain"]:
        for i in range(1, num_judges_per_category + 1):
            judge_name = f"{category}_judge_{i}"
            judges[judge_name] = JudgeConfig(
                name=judge_name,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    return TrustScoreConfig(
        span_tagger=span_tagger_config,
        judges=judges
    )


def load_sample_data(dataset_path: Optional[str] = None, num_samples: int = 50, 
                    random_seed: int = 42) -> List[Dict[str, Any]]:
    """Load sample data for profiling"""
    import random
    
    random.seed(random_seed)
    
    # Try to load from SummEval if path provided
    if dataset_path and os.path.exists(dataset_path):
        try:
            from scripts.load_summeval import load_summeval_with_sources
            all_samples = load_summeval_with_sources(dataset_path, max_samples=None)
            random.shuffle(all_samples)
            selected = all_samples[:num_samples]
            
            # Transform to TrustScore format
            samples = []
            for sample in selected:
                source_article = sample.get("source_article", "")
                summary = sample.get("summary", "")
                prompt = f"Summarize the following article:\n\n{source_article}" if source_article else "Generate a summary."
                samples.append({
                    "prompt": prompt,
                    "response": summary,
                    "model": sample.get("model_id", "unknown"),
                    "sample_id": sample.get("id", "unknown")
                })
            return samples
        except Exception as e:
            print(f"[WARNING] Could not load from dataset: {e}")
    
    # Fallback: return empty list (user should provide samples)
    print(f"[INFO] No dataset provided. Please provide sample data.")
    return []


# Cell 4: Run profiling
# Example usage:
# 
# # ============================================================================
# # CONFIGURATION (Customize these values)
# # ============================================================================
# NUM_SAMPLES = 50  # Number of responses to profile
# NUM_JUDGES_PER_CATEGORY = 3  # Judges per category (T, B, E)
# DATASET_PATH = "datasets/raw/summeval/model_annotations.aligned.jsonl"  # Optional
# API_KEY = "your-api-key"  # Required if using OpenAI provider
# USE_MOCK = False  # Set to True for testing without API calls
# GENERATION_SEED = 42  # For reproducibility
# 
# # ============================================================================
# # EXECUTION
# # ============================================================================
# 
# # Step 1: Create config with specified number of judges
# config = create_config_with_judge_count(
#     num_judges_per_category=NUM_JUDGES_PER_CATEGORY,
#     model="meta-llama/Llama-3.1-8B-Instruct",  # Default: LLaMA, change if needed
#     provider=LLMProvider.LLAMA,  # Default: LLaMA, change if needed
#     temperature=0.1,  # Model hyperparameters
#     max_tokens=2000
# )
# 
# # Step 2: Load sample data
# samples = load_sample_data(
#     dataset_path=DATASET_PATH,
#     num_samples=NUM_SAMPLES,
#     random_seed=GENERATION_SEED
# )
# 
# if not samples:
#     print("[ERROR] No samples loaded. Please provide sample data.")
#     print("Samples should be a list of dicts with 'prompt' and 'response' keys.")
# else:
#     # Step 3: Create profiler
#     profiler = PipelineProfiler(
#         config=config,
#         api_key=API_KEY,
#         use_mock=USE_MOCK
#     )
#     
#     # Step 4: Run profiling
#     print(f"\n[Profiling] Processing {len(samples)} samples with {NUM_JUDGES_PER_CATEGORY} judges per category...")
#     profiler.profile_batch(samples, generation_seed=GENERATION_SEED)
#     
#     # Step 5: Generate outputs
#     config_name = f"J{NUM_JUDGES_PER_CATEGORY}"
#     profiler._generate_outputs(config_name=config_name, output_dir="paper_materials")
#     
#     print(f"\n[Complete] Profiling finished!")
#     print(f"  - Results saved to: paper_materials/runtime_summary.json")
#     print(f"  - Report saved to: paper_materials/runtime_summary.md")

