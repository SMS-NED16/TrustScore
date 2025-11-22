"""
Run CI Calibration Analysis (RunPod Compatible with Runtime Patches)

This script runs TrustScore pipeline with varying numbers of judges
to evaluate confidence interval calibration.

This version:
- Patches the necessary methods at runtime with updated seed handling
- Saves all results locally (no Google Drive)
- Works even if the base codebase isn't updated
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import random
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.orchestrator import TrustScorePipeline
from config.settings import TrustScoreConfig, JudgeConfig, SpanTaggerConfig, LLMProvider, AggregationWeights
from modules.judges.base_judge import BaseJudge
from modules.llm_providers.vllm_provider import VLLMProvider
# Note: load_summeval_with_sources is imported lazily inside the function to avoid early CUDA initialization
from specificity_analysis.dual_logger import DualLogger, initialize_logging, cleanup_logging


# ============================================================================
# RUNTIME PATCHES - Apply updated seed handling code
# ============================================================================

def patch_batch_analyze_spans():
    """Patch BaseJudge.batch_analyze_spans to support per-item seeds."""
    original_method = BaseJudge.batch_analyze_spans
    
    def patched_batch_analyze_spans(self, span_records, seed=None, seeds=None):
        """
        Analyze multiple spans using batch processing for efficiency.
        
        Args:
            span_records: List of (LLMRecord, SpanTag) tuples to analyze
            seed: Single seed for all spans (for backward compatibility)
            seeds: List of seeds, one per span (takes precedence over seed)
        """
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider {self.config.provider} not available")
        
        # Create user prompts for all spans
        user_prompts = []
        for llm_record, span in span_records:
            user_prompt = self._create_user_prompt(llm_record, span)
            user_prompts.append(user_prompt)
        
        # Batch generate using the provider
        messages_list = [
            self.llm_provider.format_messages(self.system_prompt, prompt)
            for prompt in user_prompts
        ]
        
        try:
            # Use batch_generate if available
            if hasattr(self.llm_provider, 'batch_generate'):
                # If seeds list provided, use it; otherwise fall back to single seed
                if seeds is not None:
                    contents = self.llm_provider.batch_generate(messages_list, seeds=seeds)
                else:
                    contents = self.llm_provider.batch_generate(messages_list, seed=seed)
            else:
                # Fallback to individual calls
                contents = []
                for i, messages in enumerate(messages_list):
                    item_seed = seeds[i] if seeds is not None else seed
                    if item_seed is not None:
                        content = self.llm_provider.generate(messages, seed=item_seed)
                    else:
                        content = self.llm_provider.generate(messages)
                    contents.append(content)
            
            # Parse all responses
            analyses = []
            for content in contents:
                analysis_data = self._parse_response(content)
                analysis = self._create_judge_analysis(analysis_data)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            raise RuntimeError(f"Error in batch judge analysis: {str(e)}")
    
    BaseJudge.batch_analyze_spans = patched_batch_analyze_spans
    print("✓ Patched BaseJudge.batch_analyze_spans with per-item seed support")


def patch_batch_generate():
    """Patch VLLMProvider.batch_generate to handle per-item seeds."""
    from vllm import SamplingParams
    
    original_method = VLLMProvider.batch_generate
    
    def patched_batch_generate(self, messages_list, seed=None, seeds=None, **kwargs):
        """
        Generate responses for multiple message sets using batched inference.
        
        Args:
            messages_list: List of message sets
            seed: Single seed for all prompts (for backward compatibility)
            seeds: List of seeds, one per prompt (takes precedence over seed)
            **kwargs: Additional generation parameters
        """
        if not self.llm:
            raise ValueError("vLLM model not loaded")
        
        # Convert all messages to prompts
        prompts = [self._format_messages_to_prompt(messages) for messages in messages_list]
        
        # Get temperature
        temp = kwargs.get('temperature', self.config.temperature if hasattr(self.config, 'temperature') else 0.0)
        
        # DEBUG: Log what we're receiving to diagnose why per-item seeds might not be used
        print(f"[DEBUG vLLM] batch_generate called: seeds={seeds}, seed={seed}, temp={temp}, len(prompts)={len(prompts)}")
        if seeds is not None:
            print(f"[DEBUG vLLM] seeds check: len(seeds)={len(seeds)}, len(prompts)={len(prompts)}, temp={temp}, temp>0={temp > 0}, condition_met={seeds is not None and len(seeds) == len(prompts) and temp > 0}")
        
        # If we have per-item seeds and temperature > 0, generate separately
        # This ensures each prompt gets its own seed for variability
        if seeds is not None and len(seeds) == len(prompts) and temp > 0:
            debug_msg = f"[DEBUG vLLM] Using per-item seeds: {seeds} for {len(prompts)} prompts"
            print(debug_msg)  # DualLogger will capture this to file automatically
            # Generate each prompt separately with its unique seed
            results = []
            for i, (prompt, item_seed) in enumerate(zip(prompts, seeds)):
                sampling_params = SamplingParams(
                    temperature=temp,
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens if hasattr(self.config, 'max_tokens') else 4096),
                    top_p=kwargs.get('top_p', 0.95),
                    seed=item_seed,  # Unique seed per prompt
                )
                outputs = self.llm.generate([prompt], sampling_params)
                result_text = outputs[0].outputs[0].text.strip()
                results.append(result_text)
                print(f"[DEBUG vLLM] Prompt {i} with seed {item_seed}: generated {len(result_text)} chars")
            return results
        
        # Otherwise, use batch generation with single seed (or None)
        print(f"[DEBUG vLLM] Using batch generation (fallback): seed={seed}, temp={temp}, reason: seeds={seeds is None}, len_match={seeds is None or len(seeds) != len(prompts) if seeds is not None else 'N/A'}, temp_ok={temp > 0}")
        sampling_params = self.sampling_params
        if kwargs or seed is not None:
            if temp > 0:
                final_seed = seed if seed is not None else None
            elif 'seed' in kwargs:
                final_seed = kwargs.get('seed')
            else:
                final_seed = 42  # Default deterministic seed for temp=0
            
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens if hasattr(self.config, 'max_tokens') else 4096),
                top_p=kwargs.get('top_p', 0.95 if temp > 0 else 1.0),
                seed=final_seed,
            )
        
        # Batch generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text from outputs
        return [output.outputs[0].text.strip() for output in outputs]
    
    VLLMProvider.batch_generate = patched_batch_generate
    print("✓ Patched VLLMProvider.batch_generate with per-item seed support")


def patch_grade_spans_with_batching():
    """Patch TrustScorePipeline._grade_spans_with_batching to derive per-item seeds."""
    original_method = TrustScorePipeline._grade_spans_with_batching
    
    def patched_grade_spans_with_batching(self, llm_record, span_list, aspect_judges, 
                                         error_type, graded_spans, ensemble_config, 
                                         error_config, generation_seed=None):
        """Grade spans using batching (optimized for vLLM)"""
        import random as rng
        
        print(f"[DEBUG Judge Calls] Using BATCHING for {error_type} spans (vLLM optimization)")
        
        # Process each judge separately (each judge batches all spans)
        for judge_idx, (judge_name, judge) in enumerate(aspect_judges.items()):
            try:
                # Prepare span records for batching: (LLMRecord, SpanTag) tuples
                span_records = [(llm_record, span) for _, span in span_list]
                
                # Derive unique seeds per judge AND per span item
                # Formula: base_seed + (judge_idx * 10000) + (span_idx * 100)
                if generation_seed is not None:
                    # Derive unique seed for each span in the batch (reproducible)
                    span_seeds = [
                        generation_seed + (judge_idx * 10000) + (span_idx * 100)
                        for span_idx in range(len(span_list))
                    ]
                    debug_msg = f"[DEBUG Seeds] Judge {judge_idx}, generation_seed={generation_seed}, span_seeds={span_seeds}"
                    print(debug_msg)  # DualLogger will capture this to file automatically
                    # Batch analyze all spans for this judge with unique seeds per span
                    analyses = judge.batch_analyze_spans(span_records, seeds=span_seeds)
                else:
                    # Backward compatibility: use single random seed per judge (old behavior)
                    judge_seed = rng.randint(1, 2**31 - 1)  # Random seed per judge
                    analyses = judge.batch_analyze_spans(span_records, seed=judge_seed)
                
                # Assign analyses to corresponding spans
                for (span_id, span), analysis in zip(span_list, analyses):
                    # Get or create graded span
                    if span_id not in graded_spans.spans:
                        graded_span = GradedSpan(
                            start=span.start,
                            end=span.end,
                            type=span.type,
                            subtype=span.subtype,
                            explanation=span.explanation
                        )
                        graded_spans.spans[span_id] = graded_span
                    else:
                        graded_span = graded_spans.spans[span_id]
                    
                    # Add judge analysis
                    graded_span.add_judge_analysis(judge_name, analysis)
                    print(f"[DEBUG Judge Calls] ✓ Judge '{judge_name}' batched analysis for span {span_id}")
                
            except Exception as e:
                print(f"[ERROR Judge Calls] ✗ Judge '{judge_name}' FAILED in batch: {str(e)}")
                import traceback
                print(f"[ERROR Judge Calls]   Traceback: {traceback.format_exc()}")
                if error_config.fail_fast:
                    if error_config.continue_on_span_errors:
                        continue
                    else:
                        raise
        
        # Validate and add spans
        for span_id, span in span_list:
            if span_id in graded_spans.spans:
                graded_span = graded_spans.spans[span_id]
                successful_analyses = len(graded_span.analysis)
                
                if successful_analyses >= ensemble_config.min_judges_required:
                    if ensemble_config.require_consensus:
                        if self._check_consensus(graded_span, ensemble_config.consensus_threshold):
                            graded_spans.add_graded_span(span_id, graded_span)
                        else:
                            print(f"[DEBUG Judge Calls] Span {span_id} NOT added (consensus failed)")
                    else:
                        graded_spans.add_graded_span(span_id, graded_span)
                        print(f"[DEBUG Judge Calls] Span {span_id} added to graded_spans")
                elif not error_config.continue_on_span_errors:
                    raise RuntimeError(f"Insufficient judge analyses: {successful_analyses} < {ensemble_config.min_judges_required}")
                else:
                    print(f"[WARNING Judge Calls] Span {span_id} NOT added: insufficient analyses "
                          f"({successful_analyses} < {ensemble_config.min_judges_required})")
    
    # Import GradedSpan for the patch
    from pipeline.orchestrator import GradedSpan
    
    TrustScorePipeline._grade_spans_with_batching = patched_grade_spans_with_batching
    print("✓ Patched TrustScorePipeline._grade_spans_with_batching with per-item seed derivation")


# Apply all patches
print("Applying runtime patches for per-item seed handling...")
patch_batch_analyze_spans()
patch_batch_generate()
patch_grade_spans_with_batching()
print("✓ All patches applied successfully!\n")


# ============================================================================
# CI CALIBRATION ANALYSIS CODE
# ============================================================================

# Configuration
NUM_EXAMPLES = 5  # 3-5 examples
JUDGE_COUNTS = [1, 3, 5]  # Number of trustworthiness judges
NUM_REPEATS = 5  # Repeats per (example, J) combination
SPAN_TAGGER_TEMPERATURE = 0.0  # Deterministic for consistent spans
JUDGE_TEMPERATURE = 0.7  # Stochastic for variability
CONFIDENCE_LEVEL = 0.95  # 95% CI

# Model configuration
VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 4096  # Increased to ensure complete JSON responses (consistent with sensitivity/specificity analyses)
DEVICE = "cuda"


def create_config_for_judge_count(num_judges: int) -> TrustScoreConfig:
    """
    Create TrustScoreConfig with specified number of trustworthiness judges.
    Uses same model for all judges (VLLM limitation).
    
    Args:
        num_judges: Number of trustworthiness judges (1, 3, or 5)
        
    Returns:
        TrustScoreConfig instance
    """
    # Span Tagger Config with VLLM (temperature = 0.0 for consistency)
    span_tagger_config = SpanTaggerConfig(
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        model_path=VLLM_MODEL,
        temperature=SPAN_TAGGER_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
        torch_dtype="float16"
    )
    
    # Judge Configs with VLLM - Use same model for all judges
    judge_configs = {}
    
    # Create judge configs
    for i in range(1, num_judges + 1):
        judge_configs[f"trust_judge_{i}"] = JudgeConfig(
            name=f"trust_judge_{i}",
            enabled=True,
            provider=LLMProvider.VLLM,
            model=VLLM_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    
    # Disabled judges for pipeline compatibility
    judge_configs["bias_judge_1"] = JudgeConfig(
        name="bias_judge_1",
        enabled=False,
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    judge_configs["explainability_judge"] = JudgeConfig(
        name="explainability_judge",
        enabled=False,
        provider=LLMProvider.VLLM,
        model=VLLM_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Create full config
    config = TrustScoreConfig(
        span_tagger=span_tagger_config,
        judges=judge_configs,
        aggregation_weights=AggregationWeights(
            trustworthiness=0.6,
            explainability=0.3,
            bias=0.1
        ),
        confidence_level=CONFIDENCE_LEVEL
    )
    
    return config


def format_result_for_storage(result, sample: Dict[str, Any], run_id: str, num_judges: int, repeat: int) -> Dict[str, Any]:
    """
    Format pipeline result for storage.
    
    Args:
        result: AggregatedOutput from pipeline
        sample: Original sample data
        run_id: Unique run identifier
        num_judges: Number of judges used
        repeat: Repeat number (1-indexed)
        
    Returns:
        Formatted result dictionary
    """
    # Helper to format CI
    def format_ci(ci):
        if ci is None:
            return None
        return {
            "lower": ci.lower,
            "upper": ci.upper
        }
    
    # Extract all CI data at different levels
    result_data = {
        "item_id": sample.get("unique_dataset_id", f"{sample.get('sample_id', 'unknown')}-{sample.get('model', 'unknown')}"),
        "sample_id": sample.get("sample_id", "unknown"),
        "model_id": sample.get("model", "unknown"),
        "run_id": run_id,
        "num_judges": num_judges,
        "repeat": repeat,
        "confidence_level": CONFIDENCE_LEVEL,
        
        # Final TrustScore CIs
        "trust_score": result.summary.trust_score,
        "trust_score_ci": format_ci(result.summary.trust_score_ci),
        "trust_quality": result.summary.trust_quality,
        "trust_quality_ci": format_ci(result.summary.trust_quality_ci),
        "trust_confidence": result.summary.trust_confidence,
        "trust_confidence_ci": format_ci(result.summary.trust_confidence_ci),
        
        # Category-level CIs (T only, since E and B will be 0)
        "agg_score_T": result.summary.agg_score_T,
        "agg_score_T_ci": format_ci(result.summary.agg_score_T_ci),
        "agg_quality_T": result.summary.agg_quality_T,
        "agg_quality_T_ci": format_ci(result.summary.agg_quality_T_ci),
        "agg_confidence_T": result.summary.agg_confidence_T,
        "agg_confidence_T_ci": format_ci(result.summary.agg_confidence_T_ci),
        
        "agg_score_E": result.summary.agg_score_E,
        "agg_score_E_ci": format_ci(result.summary.agg_score_E_ci),
        "agg_confidence_E": result.summary.agg_confidence_E,
        "agg_confidence_E_ci": format_ci(result.summary.agg_confidence_E_ci),
        
        "agg_score_B": result.summary.agg_score_B,
        "agg_score_B_ci": format_ci(result.summary.agg_score_B_ci),
        "agg_confidence_B": result.summary.agg_confidence_B,
        "agg_confidence_B_ci": format_ci(result.summary.agg_confidence_B_ci),
        
        # Span-level CIs (for each error)
        "span_level_ci": []
    }
    
    # Add span-level CIs
    for error_id, error in result.errors.items():
        span_ci_data = {
            "error_id": error_id,
            "type": error.type.value,
            "subtype": error.subtype,
            "severity_score": error.severity_score,
            "severity_score_ci": format_ci(error.severity_score_ci),
            "confidence_level": error.confidence_level,
            "confidence_ci": format_ci(error.confidence_ci)
        }
        result_data["span_level_ci"].append(span_ci_data)
    
    # Metadata
    result_data["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "span_tagger_temperature": SPAN_TAGGER_TEMPERATURE,
        "judge_temperature": JUDGE_TEMPERATURE,
        "model": VLLM_MODEL,
        "num_spans_detected": len(result.errors)
    }
    
    return result_data


def run_ci_calibration_analysis(
    summeval_path: str,
    output_dir: str,
    num_examples: int = NUM_EXAMPLES,
    judge_counts: List[int] = JUDGE_COUNTS,
    num_repeats: int = NUM_REPEATS,
    use_mock: bool = False,
    api_key: Optional[str] = None,
    random_seed: int = 42
) -> str:
    """
    Run CI calibration analysis.
    
    Args:
        summeval_path: Path to SummEval JSONL file
        output_dir: Directory to save results
        num_examples: Number of examples to use (3-5)
        judge_counts: List of judge counts to test [1, 3, 5]
        num_repeats: Number of repeats per (example, J) combination
        use_mock: Whether to use mock components (for testing)
        api_key: API key for LLM providers
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to results directory
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_subdir = f"ci_calibration_results_{timestamp}"
    full_output_dir = os.path.join(output_dir, results_subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Initialize logging (DualLogger redirects stdout to both console and file)
    logger = initialize_logging(full_output_dir, "execution.log")
    
    try:
        print("=" * 70)
        print("CI CALIBRATION ANALYSIS (RunPod - Patched)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - Number of examples: {num_examples}")
        print(f"  - Judge counts: {judge_counts}")
        print(f"  - Repeats per (example, J): {num_repeats}")
        print(f"  - Total runs: {num_examples * len(judge_counts) * num_repeats}")
        print(f"  - Span tagger temperature: {SPAN_TAGGER_TEMPERATURE}")
        print(f"  - Judge temperature: {JUDGE_TEMPERATURE}")
        print(f"  - Confidence level: {CONFIDENCE_LEVEL}")
        print(f"  - Model: {VLLM_MODEL}")
        print(f"  - Output directory: {full_output_dir}")
        print("=" * 70)
        
        # Load SummEval data (lazy import to avoid early CUDA initialization)
        print("\nLoading SummEval data...")
        from scripts.load_summeval import load_summeval_with_sources
        all_samples = load_summeval_with_sources(summeval_path, max_samples=None)
        print(f"Loaded {len(all_samples)} samples")
        
        # Sort samples by unique_dataset_id for deterministic order
        all_samples.sort(key=lambda x: x.get("unique_dataset_id", ""))
        print("Sorted samples by unique_dataset_id for reproducibility")
        
        # Select random subset (now deterministic due to sorted order)
        selected_samples = random.sample(all_samples, min(num_examples, len(all_samples)))
        print(f"Selected {len(selected_samples)} samples for analysis")
        
        # Transform samples to TrustScore format
        for sample in selected_samples:
            source_article = sample.get("source_article", "")
            summary = sample.get("summary", "")
            
            # Create prompt from source article
            if source_article:
                sample["prompt"] = f"Summarize the following article:\n\n{source_article}"
            else:
                sample["prompt"] = "Generate a summary of the article."
            
            # Map summary to response
            sample["response"] = summary
            
            # Map model_id to model for compatibility
            sample["model"] = sample.get("model_id", "unknown")
            sample["sample_id"] = sample.get("id", "unknown")
        
        print("Transformed samples to TrustScore format (prompt/response)")
        
        # Save selected samples metadata
        samples_metadata = {
            "num_examples": len(selected_samples),
            "samples": [
                {
                    "item_id": s.get("unique_dataset_id", f"{s.get('sample_id', 'unknown')}-{s.get('model', 'unknown')}"),
                    "sample_id": s.get("sample_id", "unknown"),
                    "model_id": s.get("model", "unknown")
                }
                for s in selected_samples
            ]
        }
        metadata_path = os.path.join(full_output_dir, "samples_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(samples_metadata, f, indent=2)
        print(f"  ✓ Saved samples metadata to {metadata_path}")
        
        # Prepare results file
        results_file = os.path.join(full_output_dir, "calibration_results.jsonl")
        
        # Run experiments
        total_runs = len(selected_samples) * len(judge_counts) * num_repeats
        run_counter = 0
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for sample_idx, sample in enumerate(selected_samples, 1):
                print(f"\n{'='*70}")
                print(f"Processing Example {sample_idx}/{len(selected_samples)}")
                print(f"Item ID: {sample.get('unique_dataset_id', 'unknown')}")
                print(f"{'='*70}")
                
                for num_judges in judge_counts:
                    print(f"\n--- Config: {num_judges} judge(s) ---")
                    print(f"  Models: {num_judges}× LLaMA 3.1 8B (same model, different seeds for randomness)")
                    
                    # Create config for this judge count
                    config = create_config_for_judge_count(num_judges)
                    
                    # Initialize pipeline
                    pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
                    
                    # Note: DualLogger already captures all print() statements to file,
                    # so we don't need to store logger references
                    
                    for repeat in tqdm(range(1, num_repeats + 1), desc=f"  {num_judges} judge(s), repeats", unit="repeat"):
                        run_counter += 1
                        
                        # Generate run ID
                        run_id = f"run_{run_counter}::judges_{num_judges}::repeat_{repeat}::sample_{sample_idx}::{sample.get('unique_dataset_id', 'unknown')}"
                        
                        try:
                            # Verify required fields exist before processing
                            if "prompt" not in sample:
                                raise KeyError(f"Sample missing 'prompt' field. Available keys: {list(sample.keys())}")
                            if "response" not in sample:
                                raise KeyError(f"Sample missing 'response' field. Available keys: {list(sample.keys())}")
                            
                            # Generate unique seed for this repeat to ensure different judge outputs
                            repeat_seed = random_seed + (sample_idx * 1000) + (repeat * 100)
                            print(f"[INFO] Run {run_id}: Using generation_seed={repeat_seed}")
                            
                            # Run pipeline with generation_seed (affects judge seeds, not span tagger)
                            result = pipeline.process(
                                prompt=sample["prompt"],
                                response=sample["response"],
                                model=sample.get("model", "unknown"),
                                generated_on=datetime.now(),
                                generation_seed=repeat_seed  # Unique seed per repeat for judge variability
                            )
                            
                            # Format and save result
                            result_data = format_result_for_storage(
                                result, sample, run_id, num_judges, repeat
                            )
                            
                            f.write(json.dumps(result_data) + '\n')
                            f.flush()
                            
                            # Log key metrics for debugging
                            print(f"[INFO] Run {run_id} completed: trust_score={result_data['trust_score']:.3f}, "
                                  f"agg_score_T={result_data['agg_score_T']:.3f}, "
                                  f"num_spans={result_data['metadata']['num_spans_detected']}")
                            
                        except Exception as e:
                            error_msg = f"Error in run {run_id}: {str(e)}"
                            print(f"\n  ✗ {error_msg}")
                            import traceback
                            print(f"[ERROR] {error_msg}")
                            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                            # Save error record
                            error_data = {
                                "item_id": sample.get("unique_dataset_id", "unknown"),
                                "run_id": run_id,
                                "num_judges": num_judges,
                                "repeat": repeat,
                                "error": str(e)
                            }
                            f.write(json.dumps(error_data) + '\n')
                            f.flush()
        
        print(f"\n✓ Results saved to: {results_file}")
        
        print("\n" + "=" * 70)
        print("CI CALIBRATION ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved in: {full_output_dir}")
        print(f"\nFiles created:")
        print(f"  - samples_metadata.json")
        print(f"  - calibration_results.jsonl")
        print(f"  - execution.log")
        
        return full_output_dir
        
    finally:
        # Cleanup logging
        if logger is not None:
            cleanup_logging(logger)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # For RunPod notebook usage - detect if running in notebook
    try:
        from IPython import get_ipython
        get_ipython()
        # Running in notebook - use default parameters
        print("Running in Jupyter notebook - using default parameters")
        print("To customize, call run_ci_calibration_analysis() directly with your parameters\n")
        
        output_dir = run_ci_calibration_analysis(
            summeval_path="datasets/raw/summeval/model_annotations.aligned.jsonl",
            output_dir="results",
            num_examples=NUM_EXAMPLES,
            judge_counts=JUDGE_COUNTS,
            num_repeats=NUM_REPEATS,
            random_seed=42
        )
        
        print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    except NameError:
        # Not in notebook - use argparse for command-line interface
        import argparse
        
        parser = argparse.ArgumentParser(description="Run CI Calibration Analysis (RunPod Compatible)")
        parser.add_argument("--summeval-path", type=str, required=True,
                            help="Path to SummEval JSONL file")
        parser.add_argument("--output-dir", type=str, default="results",
                            help="Output directory (default: results)")
        parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES,
                            help=f"Number of examples (default: {NUM_EXAMPLES})")
        parser.add_argument("--judge-counts", type=int, nargs="+", default=JUDGE_COUNTS,
                            help=f"Judge counts to test (default: {JUDGE_COUNTS})")
        parser.add_argument("--num-repeats", type=int, default=NUM_REPEATS,
                            help=f"Number of repeats per (example, J) (default: {NUM_REPEATS})")
        parser.add_argument("--random-seed", type=int, default=42,
                            help="Random seed (default: 42)")
        parser.add_argument("--use-mock", action="store_true",
                            help="Use mock components (for testing)")
        
        args = parser.parse_args()
        
        # Run analysis
        run_ci_calibration_analysis(
            summeval_path=args.summeval_path,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            judge_counts=args.judge_counts,
            num_repeats=args.num_repeats,
            use_mock=args.use_mock,
            random_seed=args.random_seed
        )

