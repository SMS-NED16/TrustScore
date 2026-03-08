# MADIX Judge & Model Recommender

## Overview

The MADIX Recommender is a benchmark-informed recommendation engine for the TrustScore
pipeline. It provides two distinct recommendation flows:

1. **Recommend a Judge** — Given an evaluation domain, recommend the best LLM judge
   models by aggregating ranks across multiple leaderboard sources.
2. **Recommend a Model** — Given a specific task in a hierarchical taxonomy, surface
   the top-performing models directly from that task's benchmark leaderboard.

Both flows use **rank-only** logic. There are no raw score weights, normalization
heuristics, or learned ranking models. All recommendations are deterministic and
fully transparent via a decision trace.

---

## Flow 1: Recommend a Judge

### Purpose

When a user needs to evaluate LLM outputs for a particular domain (e.g.,
summarization, general QA, bias detection), this flow recommends which LLM to use
as a **judge** — the model that scores or grades the outputs.

### Inputs

| Parameter              | Required | Description                                                |
|------------------------|----------|------------------------------------------------------------|
| `domain`               | Yes      | One of: `summarization`, `general_qa`, `mathematical_qa`, `medical_qa`, `bias`, `explainability` |
| `top_k`                | Yes      | Number of judges to recommend (default: 3)                 |
| `evaluated_model_name` | No       | Name of the model being evaluated                          |
| `evaluated_model_family` | No     | Family of the model being evaluated (overrides auto-inference) |
| `exclude_same_family`  | No       | Whether to exclude judges from the same family as the evaluated model (default: true) |

### Rank Sources

The recommender draws from three independent rank sources:

| Source | What it measures | Leaderboard |
|--------|-----------------|-------------|
| **NVIDIA** | Judge reliability — agreement with human evaluators (Cohen's kappa) | NVIDIA Judge's Verdict benchmark |
| **ProLLM** | Judge acceptability classification accuracy | ProLLM LLM-as-a-Judge benchmark (currently no data available) |
| **Domain** | Model performance on domain-specific benchmarks relevant to the evaluation task | Multiple benchmarks (varies by domain) |

Each domain maps to a set of domain-specific benchmarks:

| Domain | Benchmarks |
|--------|-----------|
| Summarization | Vectara Hallucination Leaderboard (HHEM) |
| General QA | FACTS Suite, SimpleQA Verified, FACTS Grounding |
| Bias | BBQ Gender, StereoSet (BiasBench), CrowS-Pairs (BiasBench) |
| Explainability | ComVE Subtask B, E-KAR, ERASER FEVER |
| Mathematical QA | *(no benchmarks yet — placeholder)* |
| Medical QA | *(no benchmarks yet — placeholder)* |

### Rank Aggregation Logic

#### Step 1: Load rank records

All rank records are loaded from the three sources (NVIDIA, ProLLM, and domain
benchmark files) and model names are canonicalized to a common registry.

#### Step 2: Compute domain rank

For each model that appears in any of the domain's benchmarks:

1. Collect the model's rank from each benchmark in the domain.
2. Compute the **median** of those ranks.
3. If two models have the same median, break ties by:
   - **Higher benchmark coverage** (appeared in more benchmarks).
   - **Best individual rank** (lowest single rank across benchmarks).
4. Assign sequential domain rank positions (1, 2, 3, ...).

**Example** — Domain: `general_qa` (benchmarks: FACTS Suite, SimpleQA, FACTS Grounding):

| Model | FACTS Suite rank | SimpleQA rank | FACTS Grounding rank | Median | Coverage | Best | Domain Rank |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemini 2.5 Pro | 2 | 2 | — | 2.0 | 2 | 2 | 1 |
| Gemini 3 Pro | 1 | — | — | 1.0 | 1 | 1 | 2 |

Even though Gemini 3 Pro has a lower median (1.0 vs 2.0), when both median and
coverage differ, median takes priority: Gemini 3 Pro (median 1.0) ranks above
Gemini 2.5 Pro (median 2.0). Coverage only breaks ties when medians are equal.

#### Step 3: Build candidate pool

The union of all models appearing in any of the three sources forms the candidate
pool.

#### Step 4: Same-family exclusion

If `exclude_same_family` is enabled and the evaluated model's family is known, any
candidate judge from the same model family is removed from the pool. For example, if
the evaluated model is GPT-4o (family: `gpt`), then GPT-4 and GPT-5 would be
excluded as judge candidates.

#### Step 5: Final combined ranking (lexicographic sort)

All remaining candidates are sorted using a **lexicographic key** with the following
priority:

```
Sort key = (nvidia_rank, prollm_rank, domain_rank, -coverage_count, model_name)
```

| Priority | Field | Direction | Missing value |
|:--------:|-------|-----------|---------------|
| 1 | NVIDIA rank | Lower is better | Sorts last (infinity) |
| 2 | ProLLM rank | Lower is better | Sorts last (infinity) |
| 3 | Domain rank | Lower is better | Sorts last (infinity) |
| 4 | Coverage count | Higher is better | 0 |
| 5 | Model name | Alphabetical | — |

**Key properties:**
- Models with an NVIDIA rank always sort before models without one, regardless of
  domain rank.
- Missing ranks never penalize — they simply sort after present ranks.
- No weights, scores, or normalization are involved. The ordering is purely
  positional.
- The sort is stable and deterministic.

#### Step 6: Return top-k

The first `top_k` models from the sorted list are returned, each annotated with:
- Their final combined rank position
- Individual ranks from each source (NVIDIA, ProLLM, Domain)
- Coverage count and which sources contributed
- A decision trace documenting every step

### Output

```json
{
  "domain": "explainability",
  "top_k": 3,
  "candidate_pool_size": 12,
  "used_domain_benchmarks": ["comve_subtask_b", "ekar", "eraser_fever"],
  "recommended_models": [
    {
      "model_name": "mixtral-8x22b-instruct",
      "model_family": "mistral",
      "final_rank_position": 1,
      "nvidia_rank": 1,
      "prollm_rank": null,
      "domain_rank": null,
      "coverage_count": 1,
      "coverage_sources": ["nvidia"]
    }
  ],
  "decision_trace": [
    "Domain 'explainability' maps to benchmarks: ['comve_subtask_b', 'ekar', 'eraser_fever']",
    "Loaded 12 total rank records",
    "..."
  ]
}
```

---

## Flow 2: Recommend a Model

### Purpose

When a user wants to know which model performs best for a specific task (e.g.,
hallucination control in summarization, or gender bias detection), this flow surfaces
the benchmark leaderboard rankings directly — no aggregation, no cross-benchmark
combination.

### Inputs

| Parameter       | Required | Description                                                         |
|----------------|----------|---------------------------------------------------------------------|
| `taxonomy_path` | Yes      | Dot-delimited path in the taxonomy (e.g., `bias.qa.bbq_gender`)    |
| `top_k`         | Yes      | Number of models to return (default: 5)                             |

### Hierarchical Taxonomy

Tasks are organized in a three-level hierarchy:

```
trustworthiness
  factuality
    multi_scenario .......... FACTS Suite (FACTS Score, %, higher=better)
    short_form .............. SimpleQA (Accuracy, %, higher=better)
  grounding
    longform ................ FACTS Grounding (Factuality Score, %, higher=better)
  summarization
    hallucination ........... Vectara HHEM (Hallucination Rate, %, lower=better)
  judging
    human_agreement ......... Judge's Verdict (Cohen's kappa, higher=better)
    acceptability ........... ProLLM (Accuracy, %, higher=better)

bias
  stereotype
    stereoset ............... StereoSet/BiasBench (ICAT Overall, higher=better)
    crows_pairs ............. CrowS-Pairs/BiasBench (SS Overall, %, lower=better)
  qa
    bbq_gender .............. BBQ Gender (Bias Rate, %, lower=better)

explainability
  commonsense
    explanation_selection ... ComVE Subtask B (Accuracy, %, higher=better)
  reasoning
    analogical .............. E-KAR (Accuracy, %, higher=better)
  rationales
    extractive .............. ERASER FEVER (Token F1, higher=better)
```

The user navigates this tree through the UI (or specifies a dot-path via the API).
Each leaf node maps to exactly one benchmark.

### Logic

1. Resolve the `taxonomy_path` to a taxonomy node.
2. If the node is a **branch**, return its children for further navigation.
3. If the node is a **leaf**, load that benchmark's rank file directly and return the
   top-k models ordered by rank, with their original scores and benchmark metadata.

There is no cross-benchmark aggregation in this flow. Each benchmark's leaderboard is
presented independently.

### Output

```json
{
  "taxonomy_path": "bias.qa.bbq_gender",
  "display_name": "Bias > Bias in QA > Gender Bias in QA (BBQ)",
  "benchmark_name": "bbq_gender",
  "metric_name": "Bias Rate",
  "metric_units": "%",
  "metric_direction": "lower_better",
  "snapshot_date": "2025-04-18",
  "top_k": 3,
  "recommended_models": [
    {"model_name": "gpt-4", "model_family": "gpt", "rank": 1, "score": 1.0},
    {"model_name": "deepseek-v3", "model_family": "deepseek", "rank": 2, "score": 12.0},
    {"model_name": "llama-4", "model_family": "llama", "rank": 3, "score": 35.0}
  ]
}
```

---

## UI

The recommender is accessible at `/recommender` in the MADIX web interface. It
provides a tabbed layout:

- **Tab 1: Recommend a Judge** — Domain dropdown, top-k input, optional model
  name/family fields, same-family exclusion checkbox, and a ranked results table
  with per-source rank columns and coverage badges.
- **Tab 2: Recommend a Model** — Breadcrumb-based taxonomy navigator. Click through
  domain > sub-domain > task cards. At a leaf, the benchmark's ranked model table
  is displayed with scores and metric metadata.

The main MADIX page at `/` includes a navigation link to the recommender, and
vice versa.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/recommender` | Serve the recommender UI page |
| `GET` | `/api/recommender/taxonomy` | Return full taxonomy tree and judge-domain aliases |
| `GET` | `/api/recommender/taxonomy/resolve?path=...` | Resolve a taxonomy path to node info |
| `POST` | `/api/recommend-judges` | Judge recommendation (JSON body: `{domain, top_k, ...}`) |
| `POST` | `/api/recommend-model` | Model recommendation (JSON body: `{taxonomy_path, top_k}`) |

### Python API

```python
from recommender.service import recommend_judges, recommend_model

# Judge flow
result = recommend_judges(
    domain="general_qa",
    top_k=3,
    evaluated_model_family="gpt",
    exclude_same_family=True,
)

# Model flow
result = recommend_model(
    taxonomy_path="trustworthiness.summarization.hallucination",
    top_k=5,
)
```

---

## Design Principles

- **Rank-only**: No raw score aggregation, weighted formulas, or penalty functions.
- **Transparent**: Every recommendation includes a decision trace and per-source rank
  evidence.
- **Deterministic**: Same inputs always produce the same output (stable sort, no
  randomness).
- **Config-driven**: Domains, benchmarks, and model identities are defined in YAML
  configuration, not hardcoded in business logic.
- **Missing-data robust**: Models missing from a source are not penalized — they
  simply sort after models that have data for that source.
- **Extensible**: New benchmarks, domains, or leaderboard sources can be added by
  editing configuration and dropping in a new JSON rank file.

---

## Project Structure

```
recommender/
  __init__.py                   # Re-exports recommend_judges, recommend_model
  schemas.py                    # Pydantic models for all data structures
  config.py                     # Loads taxonomy.yaml, models.yaml, resolves paths
  models.py                     # ModelRegistry — canonical name/alias/family lookup
  canonicalization.py            # Maps raw leaderboard names to canonical model IDs
  service.py                    # Orchestrator: recommend_judges() + recommend_model()
  RECOMMENDER.md                # This document
  ingestion/
    base.py                     # Abstract adapter + shared JSON loading
    nvidia.py                   # NVIDIA Judge's Verdict adapter
    prollm.py                   # ProLLM adapter (empty-safe)
    domain_benchmarks.py        # Domain benchmark adapter (one JSON per benchmark)
  ranking/
    domain_rank.py              # Median rank computation for judge flow
    final_rank.py               # Lexicographic sort + same-family exclusion
    benchmark_lookup.py          # Direct benchmark lookup for model flow
  data/
    taxonomy.yaml               # Hierarchical taxonomy + judge-domain aliases
    models.yaml                 # Model registry (canonical names, families, aliases)
    ranks/
      nvidia_judges_verdict.json
      prollm_judge.json         # Empty placeholder
      domain/
        facts_suite.json
        facts_grounding.json
        simpleqa.json
        vectara_hallucination.json
        bbq_gender.json
        stereoset.json
        crows_pairs.json
        comve_subtask_b.json
        ekar.json
        eraser_fever.json
tests/
  recommender/
    test_config.py              # Taxonomy loading and path resolution
    test_models.py              # ModelRegistry and alias lookup
    test_canonicalization.py     # Raw name to canonical mapping
    test_ranking.py             # Domain rank, final rank, same-family filter
    test_integration.py         # End-to-end tests for both flows
ui/
  app.py                        # Flask routes (existing + recommender endpoints)
  static/
    recommender.html            # Standalone two-tab UI page
    recommender.js              # Client-side logic for both tabs
```

---

## Data Sources

All rank data is seeded from the benchmark survey in `deep-research-report.md`.
Current snapshots include top-3 models per benchmark. Richer rank files can replace
these as more leaderboard data becomes available.

| Source file | Snapshot date | Models |
|-------------|:------------:|:------:|
| nvidia_judges_verdict.json | 2025-10-10 | 3 |
| prollm_judge.json | 2026-03-07 | 0 (placeholder) |
| facts_suite.json | 2025-12-11 | 3 |
| facts_grounding.json | 2025-01-07 | 3 |
| simpleqa.json | 2026-03-07 | 3 |
| vectara_hallucination.json | 2026-03-05 | 3 |
| bbq_gender.json | 2025-04-18 | 3 |
| stereoset.json | 2022-04-04 | 3 |
| crows_pairs.json | 2022-04-04 | 3 |
| comve_subtask_b.json | 2020-12-12 | 3 |
| ekar.json | 2022-03-16 | 3 |
| eraser_fever.json | 2020-06-01 | 3 |

---

## Adding New Data

### Adding a new benchmark

1. Create a JSON file in `recommender/data/ranks/domain/` following the schema:
   ```json
   {
     "source": "domain",
     "benchmark_name": "my_benchmark",
     "metric_name": "Accuracy",
     "metric_units": "%",
     "metric_direction": "higher_better",
     "source_url": "https://...",
     "snapshot_date": "2026-01-01",
     "entries": [
       {"model": "Model Name", "rank": 1, "score": 95.0},
       {"model": "Another Model", "rank": 2, "score": 90.0}
     ]
   }
   ```
2. Add the benchmark as a leaf node in `taxonomy.yaml` under the appropriate branch.
3. If it should be used in a judge-flow domain, add its ID to the relevant
   `judge_domains` entry in `taxonomy.yaml`.
4. Add any new model names to `models.yaml` with their canonical name, family, and
   aliases so that canonicalization works correctly.

### Adding a new model to the registry

Add an entry to `recommender/data/models.yaml`:
```yaml
- canonical_name: "my-model"
  display_name: "My Model"
  family: "my_family"
  aliases:
    - "org/my-model-v1"
    - "My Model v1"
```

### Updating leaderboard data

Replace the contents of the relevant JSON file in `recommender/data/ranks/`. The
adapter will pick up the new data on the next request (after cache invalidation or
server restart).
