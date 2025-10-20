# TrustScore Pipeline

A comprehensive pipeline for evaluating trustworthiness, bias, and explainability in Large Language Model (LLM) responses.

## Overview

TrustScore is a multi-stage pipeline that:

1. **Ingests** LLM prompts and responses in a standardized format
2. **Identifies** span-level errors using a fine-tuned LLM tagger
3. **Scores** error severity using multiple specialized judges
4. **Aggregates** scores into final TrustScore with confidence intervals

## Features

- **Multi-category Error Detection**: Identifies Trustworthiness (T), Bias (B), and Explainability (E) errors
- **Span-level Analysis**: Precise character-level error identification
- **Multiple Judge Evaluation**: Uses specialized judges for different error types
- **Confidence Intervals**: Statistical uncertainty quantification
- **Configurable Weights**: Customizable aggregation of T/E/B scores
- **Batch Processing**: Support for processing multiple responses
- **Comprehensive Validation**: Input validation and error handling

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pipeline.orchestrator import analyze_llm_response

# Analyze a single LLM response
result = analyze_llm_response(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    model="GPT-4o",
    use_mock=True  # Use mock components for testing
)

print(f"Trust Score: {result.summary.trust_score}")
print(f"Errors found: {len(result.errors)}")
```

## Architecture

```
TrustScore Pipeline
├── Data Models (models/)
│   ├── LLMRecord - Standardized input/output format
│   ├── SpanTags - Error span identification
│   └── AggregatedOutput - Final results
├── Core Modules (modules/)
│   ├── SpanTagger - LLM-based error identification
│   ├── Judges - Specialized severity scoring
│   └── Aggregator - Score combination
├── Pipeline (pipeline/)
│   └── Orchestrator - Main coordination
├── Configuration (config/)
│   └── Settings - Weights and parameters
├── Prompts (prompts/)
│   └── System Prompts - Centralized prompt management
└── Utilities (utils/)
    └── Error Handling - Validation and logging
```

## Configuration

```python
from config.settings import TrustScoreConfig

config = TrustScoreConfig()
config.aggregation_weights.trustworthiness = 0.6
config.aggregation_weights.explainability = 0.3
config.aggregation_weights.bias = 0.1

pipeline = TrustScorePipeline(config=config)
```

## Error Types

### Trustworthiness (T)
- `spelling`: Spelling and grammatical errors
- `factual_error`: Incorrect facts or information
- `hallucination`: Completely fabricated information
- `inconsistency`: Internal contradictions

### Bias (B)
- `demographic_bias`: Demographic stereotyping
- `cultural_bias`: Cultural assumptions
- `gender_bias`: Gender stereotyping
- `political_bias`: Political leaning

### Explainability (E)
- `unclear_explanation`: Unclear or confusing explanation
- `missing_context`: Missing important context
- `overly_complex`: Unnecessarily complex language
- `assumption_not_stated`: Unstated assumptions

## API Reference

### Main Pipeline

```python
from pipeline.orchestrator import TrustScorePipeline

pipeline = TrustScorePipeline(api_key="your-openai-key")
result = pipeline.process(prompt, response, model)
```

### Individual Components

```python
from modules.span_tagger import SpanTagger
from modules.judges.trustworthiness_judge import TrustworthinessJudge
from modules.aggregator import Aggregator

# Use components individually
tagger = SpanTagger(config, api_key)
judge = TrustworthinessJudge(config, api_key)
aggregator = Aggregator(config)
```

## Examples

See `examples/usage.py` for comprehensive usage examples including:
- Basic usage patterns
- Pipeline configuration
- Validation and error handling
- Custom configurations

## Testing

```bash
# Run examples
python examples/usage.py

# Run with mock components (no API calls)
python examples/usage.py --mock
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub.
