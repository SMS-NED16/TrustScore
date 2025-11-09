"""
TrustScore Web UI - Flask Backend
Simple REST API wrapper around TrustScore pipeline for demo purposes
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import Dict, Any, Optional
import traceback

# Add parent directory to path to import TrustScore modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.orchestrator import TrustScorePipeline, analyze_llm_response
from config.settings import load_config, TrustScoreConfig

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='')
CORS(app)  # Enable CORS for frontend

# Global pipeline instance (can be initialized with API key)
pipeline: Optional[TrustScorePipeline] = None


def get_pipeline(use_mock: bool = False, api_key: Optional[str] = None, config: Optional[TrustScoreConfig] = None) -> TrustScorePipeline:
    """Get or create pipeline instance"""
    global pipeline
    # Always create a new pipeline if config is provided (to respect custom config)
    if config is not None:
        pipeline = TrustScorePipeline(config=config, api_key=api_key, use_mock=use_mock)
    elif pipeline is None:
        pipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
    return pipeline


@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory(str(STATIC_DIR), 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "TrustScore API"})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a single prompt/response pair
    
    Request body:
    {
        "prompt": "string",
        "response": "string",
        "model": "string (optional)",
        "use_mock": bool (optional, default: False),
        "api_key": "string (optional)"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt', '').strip()
        response = data.get('response', '').strip()
        model = data.get('model', 'unknown')
        use_mock = data.get('use_mock', False)
        api_key = data.get('api_key', os.getenv('OPENAI_API_KEY'))
        config_data = data.get('config', {})
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if not response:
            return jsonify({"error": "Response is required"}), 400
        
        # Create custom config if provided
        config = None
        if config_data:
            from config.settings import TrustScoreConfig, AggregationWeights, JudgeConfig, LLMProvider
            
            # Create aggregation weights
            weights = config_data.get('weights', {})
            aggregation_weights = AggregationWeights(
                trustworthiness=weights.get('trustworthiness', 0.5),
                explainability=weights.get('explainability', 0.3),
                bias=weights.get('bias', 0.2)
            )
            
            # Create judge configs based on counts
            judge_counts = config_data.get('judge_counts', {})
            judges = {}
            
            # Trustworthiness judges
            for i in range(judge_counts.get('trustworthiness', 3)):
                judges[f"trust_judge_{i+1}"] = JudgeConfig(
                    name=f"trust_judge_{i+1}",
                    model="gpt-4o",
                    provider=LLMProvider.OPENAI
                )
            
            # Explainability judges
            for i in range(judge_counts.get('explainability', 3)):
                judges[f"explain_judge_{i+1}"] = JudgeConfig(
                    name=f"explain_judge_{i+1}",
                    model="gpt-4o",
                    provider=LLMProvider.OPENAI
                )
            
            # Bias judges
            for i in range(judge_counts.get('bias', 3)):
                judges[f"bias_judge_{i+1}"] = JudgeConfig(
                    name=f"bias_judge_{i+1}",
                    model="gpt-4o",
                    provider=LLMProvider.OPENAI
                )
            
            # Create config
            config = TrustScoreConfig(
                aggregation_weights=aggregation_weights,
                judges=judges
            )
        
        # Get pipeline instance
        pipeline = get_pipeline(use_mock=use_mock, api_key=api_key, config=config)
        
        # Run analysis
        result = pipeline.process(prompt=prompt, response=response, model=model)
        
        # Format result for frontend
        formatted_result = format_result(result, pipeline)
        
        return jsonify({
            "success": True,
            "result": formatted_result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple prompt/response pairs
    
    Request body:
    {
        "samples": [
            {"prompt": "string", "response": "string", "model": "string (optional)"},
            ...
        ],
        "use_mock": bool (optional),
        "api_key": "string (optional)"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        samples = data.get('samples', [])
        use_mock = data.get('use_mock', False)
        api_key = data.get('api_key', os.getenv('OPENAI_API_KEY'))
        
        if not samples or not isinstance(samples, list):
            return jsonify({"error": "samples must be a non-empty list"}), 400
        
        # Get pipeline instance
        pipeline = get_pipeline(use_mock=use_mock, api_key=api_key)
        
        # Prepare batch inputs
        batch_inputs = []
        for sample in samples:
            if 'prompt' not in sample or 'response' not in sample:
                continue
            batch_inputs.append({
                'prompt': sample['prompt'],
                'response': sample['response'],
                'model': sample.get('model', 'unknown')
            })
        
        if not batch_inputs:
            return jsonify({"error": "No valid samples provided"}), 400
        
        # Run batch analysis
        results = pipeline.process_batch(batch_inputs)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if result:
                formatted_results.append(format_result(result, pipeline))
            else:
                formatted_results.append({
                    "error": "Processing failed for this sample",
                    "index": i
                })
        
        return jsonify({
            "success": True,
            "results": formatted_results,
            "total": len(formatted_results),
            "successful": sum(1 for r in formatted_results if "error" not in r)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc() if app.debug else None
        }), 500


def format_result(result, pipeline: TrustScorePipeline) -> Dict[str, Any]:
    """Format AggregatedOutput for frontend consumption"""
    
    # Get judge info for display
    judge_info_map = pipeline.get_judge_info_map()
    
    # Format errors with span information
    errors = []
    for error_id, error_summary in result.errors.items():
        error_data = {
            "id": error_id,
            "type": error_summary.type.value,
            "subtype": error_summary.subtype,
            "severity_bucket": error_summary.severity_bucket.value,
            "severity_score": round(error_summary.severity_score, 3),
            "confidence_level": round(error_summary.confidence_level, 3),
            "explanation": error_summary.explanation,
            "severity_score_ci": {
                "lower": round(error_summary.severity_score_ci.lower, 3) if error_summary.severity_score_ci.lower else None,
                "upper": round(error_summary.severity_score_ci.upper, 3) if error_summary.severity_score_ci.upper else None
            },
            "confidence_ci": {
                "lower": round(error_summary.confidence_ci.lower, 3) if error_summary.confidence_ci.lower else None,
                "upper": round(error_summary.confidence_ci.upper, 3) if error_summary.confidence_ci.upper else None
            }
        }
        
        # Add span information if available
        if result.graded_spans and error_id in result.graded_spans.spans:
            span = result.graded_spans.spans[error_id]
            error_data["span"] = {
                "start": span.start,
                "end": span.end,
                "text": result.llm_response[span.start:span.end] if span.end <= len(result.llm_response) else ""
            }
            
            # Add judge analyses if available
            if span.analysis:
                error_data["judge_analyses"] = {}
                for judge_name, analysis in span.analysis.items():
                    error_data["judge_analyses"][judge_name] = {
                        "severity_score": round(analysis.severity_score, 3),
                        "confidence": round(analysis.confidence, 3),
                        "severity_bucket": analysis.severity_bucket.value,
                        "model": judge_info_map.get(judge_name, {}).get("model", "unknown")
                    }
        
        errors.append(error_data)
    
    # Format summary
    summary = {
        "trust_score": round(result.summary.trust_score, 3),
        "trust_score_ci": {
            "lower": round(result.summary.trust_score_ci.lower, 3) if result.summary.trust_score_ci.lower else None,
            "upper": round(result.summary.trust_score_ci.upper, 3) if result.summary.trust_score_ci.upper else None
        },
        "trust_confidence": round(result.summary.trust_confidence, 3),
        "trust_confidence_ci": {
            "lower": round(result.summary.trust_confidence_ci.lower, 3) if result.summary.trust_confidence_ci.lower else None,
            "upper": round(result.summary.trust_confidence_ci.upper, 3) if result.summary.trust_confidence_ci.upper else None
        },
        "categories": {
            "trustworthiness": {
                "score": round(result.summary.agg_score_T, 3),
                "confidence": round(result.summary.agg_confidence_T, 3),
                "score_ci": {
                    "lower": round(result.summary.agg_score_T_ci.lower, 3) if result.summary.agg_score_T_ci.lower else None,
                    "upper": round(result.summary.agg_score_T_ci.upper, 3) if result.summary.agg_score_T_ci.upper else None
                }
            },
            "explainability": {
                "score": round(result.summary.agg_score_E, 3),
                "confidence": round(result.summary.agg_confidence_E, 3),
                "score_ci": {
                    "lower": round(result.summary.agg_score_E_ci.lower, 3) if result.summary.agg_score_E_ci.lower else None,
                    "upper": round(result.summary.agg_score_E_ci.upper, 3) if result.summary.agg_score_E_ci.upper else None
                }
            },
            "bias": {
                "score": round(result.summary.agg_score_B, 3),
                "confidence": round(result.summary.agg_confidence_B, 3),
                "score_ci": {
                    "lower": round(result.summary.agg_score_B_ci.lower, 3) if result.summary.agg_score_B_ci.lower else None,
                    "upper": round(result.summary.agg_score_B_ci.upper, 3) if result.summary.agg_score_B_ci.upper else None
                }
            }
        }
    }
    
    return {
        "prompt": result.task_prompt,
        "response": result.llm_response,
        "model": result.model_metadata.model,
        "generated_on": result.model_metadata.generated_on.isoformat() if result.model_metadata.generated_on else None,
        "summary": summary,
        "errors": errors,
        "error_count": len(errors)
    }


if __name__ == '__main__':
    # Run in debug mode for development
    print("=" * 60)
    print("TrustScore Web UI Starting...")
    print("=" * 60)
    print(f"Server will be available at: http://localhost:5000")
    print(f"Static files directory: {STATIC_DIR}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

