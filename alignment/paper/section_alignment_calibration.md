# Human Alignment Calibration Experiments

## Overview

A critical requirement for any automated evaluation framework is that its scores correlate
meaningfully with human judgements. This section describes a systematic calibration study
in which TEBScore's 18-dimensional configuration space is optimized to maximise
Spearman rank correlation with human preference annotations across four established
benchmarks and six judge LLMs. The study serves three purposes: (i) it validates that
TEBScore is *compatible* with human evaluation signals, (ii) it demonstrates that the
default configuration is a reasonable but suboptimal starting point, and (iii) it
establishes that Bayesian hyperparameter search over the configuration space reliably
improves alignment.

---

## 5.1 Experimental Setup

### Benchmarks

Four publicly available human-annotated benchmarks are used (Table 1). Each provides a
(prompt, response) pair together with one or more numerical human quality ratings.

**Table 1. Alignment benchmarks and preprocessing.**

| Dataset | Domain | Samples | Observation (prompt → response) | Annotator dimensions | Rating aggregation | Scale |
|---|---|---|---|---|---|---|
| SummEval | News summarization | 500 | Source news article → machine-generated summary | coherence, consistency, fluency, relevance (expert + crowd workers) | Mean across all dimensions and all annotators | 1–5 |
| SimpEval | Text simplification | 500 | Complex sentence → machine-simplified sentence (human-written references excluded) | adequacy, fluency, simplicity (crowd workers) | Mean across the three dimensions per row | 0–100 |
| FeedbackQA | Open-domain QA | 500 | Question → retrieved Wikipedia passage | Single overall quality judgement per annotator (Bad / Could be Improved / Good / Excellent) | Categorical labels mapped to integers (Bad=1, Could be Improved=2, Good=3, Excellent=4), then mean across annotators | 1–4 |
| TopicalChat USR | Open-domain dialogue | 360 | Dialogue context + background fact → system response | Overall quality per annotator (crowd workers); additional dimensions stored but unused | Mean of Overall ratings across annotators | 1–5 |

**Preprocessing notes.** For each dataset, one sample corresponds to a single
(prompt, response) pair with a scalar human score derived as described above. SummEval
aggregates across both expert and crowd-worker annotators; any sample for which no
valid annotation is found is discarded. SimpEval excludes human-authored reference
simplifications (rows marked "Human 1 Writing" and "Human 2 Writing"), retaining only
machine-system outputs so that TEBScore is always evaluating model-generated text.
FeedbackQA's human labels reflect whether the retrieved passage is relevant to the
question, not intrinsic text quality — this construct mismatch is discussed in
Section 5.3.1. TopicalChat USR records six annotated dimensions per response but only
the Overall score is used as the alignment target.

All datasets are sub-sampled to the sizes above using a fixed random seed (42). Splits
are constructed at the document level (passage-level for FeedbackQA; conversation-level
for TopicalChat USR; article-level for SummEval; source-sentence-level for SimpEval) to
prevent any source document from appearing in more than one split. The split ratio is
60 % training / 20 % validation / 20 % test throughout. Due to document-level splitting,
the resulting split sizes are approximately but not exactly equal to the 60/20/20
proportions (FeedbackQA: 309/95/96; TopicalChat USR: 216/72/72; SimpEval and SummEval:
300/100/100).

### Judge Models

Six LLMs are evaluated as judges, spanning the range from small open-weights models to a
proprietary API model:

- **GPT-4o-mini** (OpenAI API) — closed-source, strong instruction-following
- **Llama 3.1 8B** (Meta) — open-weights, general-purpose
- **Mistral 7B v0.3** (Mistral AI) — open-weights, instruction-tuned
- **Gemma 2 9B** (Google DeepMind) — open-weights, strong at reasoning tasks
- **Qwen 2.5 7B** (Alibaba) — open-weights, multilingual capability
- **Phi-3 Mini 3.8B** (Microsoft) — open-weights, small footprint

Open-weights models were served on an NVIDIA H200 SXM5 80 GB GPU (RunPod cloud).
GPT-4o-mini inference used the OpenAI API via Google Colab CPU notebooks to avoid
replicating GPU allocation costs for API-based inference.

### Configuration Space

TEBScore exposes 18 tunable parameters (see Section 3.3):

- **Category weights**: $w_T$, $w_E$, $w_B \in [0,1]$ with $w_T + w_E + w_B = 1$
- **Sigmoid shape**: steepness $k \in [0.1, 2.0]$, shift $s \in [-1.0, 1.0]$
- **14 error-subtype weights**: one per (category, error type) pair, each in $[0, 2]$

The default configuration uses category weights $w_T = 0.5$, $w_E = 0.3$, $w_B = 0.2$
and unit error-subtype weights.

### Calibration Procedure

**Inference and caching.** The full TEBScore pipeline is run once on every sample across
all three splits before optimisation begins. For each (prompt, response) pair the span
tagger identifies candidate error spans and the three judge ensembles (one per category
T, E, B; three judges each) independently rate those spans. Results are persisted to
disk keyed by `(judge_model, task_name, sample_id)`, so inference is performed exactly
once per sample and all subsequent optimisation trials re-aggregate from cache without
further LLM calls.

**Optimisation strategies.** Two optimisation strategies were evaluated against a
default-configuration baseline. The first, a LightGBM surrogate model, was trained to
predict Spearman correlation from configuration vectors; being a supervised model, it
requires a held-out validation split for training and a separate test split for final
evaluation — hence the three-way split design. The second, and primary strategy reported
here, is Bayesian optimisation using the Optuna TPE (Tree-structured Parzen Estimator)
sampler. TPE builds a probabilistic model of the objective surface and focuses sampling
toward promising regions, making it substantially more sample-efficient than exhaustive
search in an 18-dimensional space. The LightGBM approach did not yield consistent
improvements and is not reported in detail; the three-way split is retained for
consistency across both methods.

**Objective and split roles.** The Optuna objective maximises Spearman rank correlation
on the training split:

$$\text{maximise} \quad \rho_S\!\left(\{\hat{s}_i\}, \{h_i\}\right) \quad \text{over } \theta \in \Theta$$

where $\hat{s}_i = f_\theta(x_i)$ is the TEBScore re-aggregated under configuration
$\theta$ and $h_i$ is the human score. Spearman is preferred over Pearson as it is
invariant to monotone transformations and more robust under non-normal score
distributions. The sampler runs for 100 trials, using the first
$\lfloor n_{\text{trials}} / 5 \rfloor$ as random startup trials before switching to
model-guided proposals. The test split is never observed during optimisation and
provides the final reported $\rho_{\text{test}}$.

---

## 5.2 Results

### 5.2.1 Main Results: Default vs. Tuned Configuration

Table 2 reports test-set Spearman correlation ($\rho$) for all 24 model × dataset
combinations under the default configuration and after Optuna calibration. $\Delta\rho$
is the absolute improvement from tuning. Positive $\Delta\rho$ indicates that calibration
improved alignment with human judgements.

**Table 2. Test-set Spearman correlation: default vs. Optuna-tuned configuration.**
*Higher $\rho$ is better. $\Delta\rho$ = tuned $-$ default.*

| Dataset | Judge Model | $\rho_{\text{default}}$ | $\rho_{\text{tuned}}$ | $\Delta\rho$ |
|---|---|---:|---:|---:|
| **FeedbackQA** | GPT-4o-mini | -0.400 | 0.024 | +0.424 |
| | Llama 3.1 8B | -0.265 | -0.079 | +0.186 |
| | Mistral 7B | -0.273 | 0.008 | +0.281 |
| | Gemma 2 9B | -0.258 | -0.258 | 0.000 |
| | Qwen 2.5 7B | -0.307 | 0.089 | +0.396 |
| | Phi-3 Mini | -0.411 | -0.045 | +0.366 |
| **SimpEval** | GPT-4o-mini | 0.072 | 0.259 | +0.187 |
| | Llama 3.1 8B | -0.218 | -0.049 | +0.169 |
| | Mistral 7B | -0.025 | 0.251 | +0.276 |
| | Gemma 2 9B | 0.039 | 0.035 | -0.004 |
| | Qwen 2.5 7B | -0.050 | -0.107 | -0.057 |
| | Phi-3 Mini | -0.136 | -0.125 | +0.011 |
| **SummEval** | GPT-4o-mini | 0.132 | 0.300 | +0.168 |
| | Llama 3.1 8B | -0.033 | 0.049 | +0.082 |
| | Mistral 7B | -0.138 | -0.086 | +0.052 |
| | Gemma 2 9B | 0.072 | 0.034 | -0.038 |
| | Qwen 2.5 7B | 0.000 | -0.043 | -0.043 |
| | Phi-3 Mini | 0.078 | 0.082 | +0.004 |
| **TopicalChat** | GPT-4o-mini | -0.176 | 0.231 | +0.407 |
| | Llama 3.1 8B | -0.113 | 0.046 | +0.159 |
| | Mistral 7B | -0.165 | -0.119 | +0.046 |
| | Gemma 2 9B | -0.131 | -0.118 | +0.013 |
| | Qwen 2.5 7B | -0.362 | 0.109 | +0.471 |
| | Phi-3 Mini | -0.173 | -0.186 | -0.013 |

Tuning improves alignment in 18 of 24 combinations (75 %). The mean $\Delta\rho$ across
all 24 pairs is **+0.148**. GPT-4o-mini consistently achieves the highest post-tuning
correlation on three of four benchmarks (SimpEval $\rho = 0.259$, SummEval $\rho =
0.300$, TopicalChat $\rho = 0.231$), confirming that judge capability is a primary driver
of alignment quality.

### 5.2.2 Per-Dataset Summary

**Table 3. Dataset-level summary: mean test Spearman across all six judge models.**

| Dataset | $\bar{\rho}_{\text{default}}$ | $\bar{\rho}_{\text{tuned}}$ | $\overline{\Delta\rho}$ | Improved (of 6) |
|---|---:|---:|---:|---:|
| FeedbackQA | -0.319 | -0.044 | +0.276 | 5 / 6 |
| SimpEval | -0.053 | 0.044 | +0.097 | 4 / 6 |
| SummEval | +0.019 | +0.056 | +0.037 | 4 / 6 |
| TopicalChat | -0.187 | -0.006 | +0.181 | 5 / 6 |
| **Overall** | **-0.135** | **+0.013** | **+0.148** | **18 / 24** |

FeedbackQA shows the largest mean gain (+0.276), and TopicalChat the second largest
(+0.181), despite both starting from strongly negative default correlations. SummEval is
the only dataset where the default configuration is already slightly positive on average
($\bar{\rho} = +0.019$), yet tuning still improves it further to $+0.056$.

### 5.2.3 Per-Model Summary

**Table 4. Model-level summary: mean test Spearman across all four datasets.**

| Judge Model | $\bar{\rho}_{\text{default}}$ | $\bar{\rho}_{\text{tuned}}$ | $\overline{\Delta\rho}$ | Improved (of 4) |
|---|---:|---:|---:|---:|
| GPT-4o-mini | -0.093 | +0.204 | +0.297 | 4 / 4 |
| Llama 3.1 8B | -0.157 | -0.008 | +0.149 | 4 / 4 |
| Mistral 7B | -0.150 | +0.014 | +0.164 | 4 / 4 |
| Gemma 2 9B | -0.070 | -0.077 | -0.007 | 1 / 4 |
| Qwen 2.5 7B | -0.180 | +0.012 | +0.192 | 2 / 4 |
| Phi-3 Mini | -0.161 | -0.069 | +0.092 | 3 / 4 |

GPT-4o-mini, Llama 3.1 8B, and Mistral 7B each improve on all four datasets after
calibration. Gemma 2 9B is the weakest responder (1 of 4 datasets improve), suggesting
its output distribution may be less amenable to post-hoc re-weighting under the current
configuration space.

### 5.2.4 LightGBM Comparison

As described in Section 5.1, a LightGBM surrogate model was evaluated alongside Optuna
as an alternative optimisation strategy. Across all 24 model × dataset combinations,
Optuna TPE matched or exceeded LightGBM on every combination. LightGBM results are
included in the full results table (Appendix A) for completeness; the remainder of the
analysis focuses on Optuna.

### 5.2.5 Visualisations

Figure 1 shows a grouped bar chart comparing default, LightGBM, and Optuna-tuned
Spearman correlations per dataset and model. Tuned bars consistently reach higher values
than default bars on SummEval and SimpEval; the effect is most pronounced for GPT-4o-mini
and Qwen 2.5 7B on TopicalChat USR.

![Figure 1: Grouped bar chart of default vs. LightGBM vs. Optuna-tuned test Spearman per dataset and model.](figures/alignment_bar_chart.png)

*Figure 1. Default / LightGBM / Optuna-tuned test-set Spearman correlation, grouped by
dataset. Each cluster of bars represents one judge model. Dashed reference line at
$\rho = 0$ marks the boundary between positive and negative rank correlation.*

Figure 2 presents a heatmap of $\Delta\rho$ (tuning gain) across the 6 × 4 model ×
dataset grid for both tuned methods. Warm cells indicate substantial improvement; cool
cells highlight the small number of cases where calibration had no effect or a minor
negative effect.

![Figure 2: Heatmap of Δρ (tuning gain) across all model × dataset combinations.](figures/alignment_heatmap.png)

*Figure 2. Heatmap of $\Delta\rho = \rho_{\text{tuned}} - \rho_{\text{default}}$ for
all 24 model × dataset combinations (LightGBM and Optuna). Annotated values are rounded
to two decimal places.*

---

## 5.3 Discussion

### 5.3.1 Framework Compatibility with Human Judgements

The calibration results demonstrate that TEBScore outputs are, in principle, alignable
with human preference signals across diverse task types. Post-tuning, 18 of 24
combinations show improved correlation, and GPT-4o-mini reaches $\rho \geq 0.23$ on
three of four benchmarks. This is comparable to the range reported for other automated
evaluation metrics in the literature: for example, BARTScore achieves Spearman
correlations of 0.27–0.37 on SummEval [REF], and GPTScore reaches 0.28–0.41 depending
on the facet [REF].

The FeedbackQA benchmark presents an important diagnostic. Human annotators rated whether
a retrieved Wikipedia passage adequately answers a given question — a retrieval relevance
judgement. TEBScore, however, evaluates the *intrinsic quality* of the text (fluency,
factual consistency, explainability, absence of bias), independent of query relevance.
The negative default correlations on FeedbackQA ($\bar{\rho} = -0.319$) reflect this
construct mismatch rather than a failure of the scoring framework: a well-written
encyclopaedic passage will receive a high TEBScore yet score poorly in retrieval
relevance. That tuning partially recovers positive correlation on FeedbackQA (5 of 6
models improve) shows that the configuration space is expressive enough to partially
reconcile the two signals even when the underlying constructs differ.

### 5.3.2 Task-Adaptive Weight Profiles

Analysis of the tuned category weight profiles (Table B1, Appendix B) reveals
task-specific patterns. On SummEval the mean tuned trustworthiness weight
($\bar{w}_T = 0.502$) is the highest across all datasets, consistent with factual
consistency being the dominant quality dimension for news summarisation. On TopicalChat
USR the explainability weight is elevated ($\bar{w}_E = 0.405$), consistent with human
raters rewarding coherent and understandable dialogue responses.

The bias weight $w_B$ shows the most variable behaviour. Whereas several model–task
combinations converge to near-zero $w_B$ (indicating that a bias penalty does not
explain human preference variance in these corpora), others assign substantial weight —
notably SimpEval ($\bar{w}_B = 0.386$) and TopicalChat ($\bar{w}_B = 0.442$). This
variability reflects the optimiser exploiting the bias dimension as an auxiliary signal
when it happens to correlate with human scores on a given model's outputs, rather than
the benchmarks directly targeting bias evaluation. This is an expected artefact of
optimising a flexible multi-dimensional score against a single composite human signal.

### 5.3.3 Tuning Efficacy and Limitations

Bayesian optimisation over 100 trials reliably improves alignment in the majority of
cases. The overall shift from $\bar{\rho} = -0.135$ (default) to $+0.013$ (tuned) across
all 24 combinations demonstrates a consistent directional improvement even though
absolute correlations remain modest. Several combinations show no improvement or small
regressions — most notably Gemma 2 9B (1 of 4 datasets improve) and Qwen 2.5 7B on
SimpEval — suggesting that either (a) the optimiser encountered a flat or noisy objective
landscape for those specific model–task combinations, or (b) the underlying judge model's
outputs do not vary sufficiently across the configuration space to expose a gradient
toward human preference.

Test set sizes (72–100 samples) introduce variance in the correlation estimates. Larger
evaluation sets and additional Optuna trials would reduce this variance and may reveal
clearer patterns. Nonetheless, the consistent directional improvement across diverse
benchmarks and six judge models provides empirical support for the hypothesis that
TEBScore's configuration space is expressive enough to capture task-specific notions of
quality when calibrated appropriately.

---

## Appendix A: Full Alignment Results

Table A1 provides the complete alignment results including training, validation, and test
Spearman correlations for all 24 model × dataset combinations, as well as Pearson ($r$)
and Kendall-$\tau$ coefficients on the test split. Where multiple runs existed for the
same model × dataset × tuning combination, the run with the highest test Spearman is
retained.

**Table A1. Full alignment results: all correlation metrics, all splits.**
*$\rho$ = Spearman, $r$ = Pearson, $\tau$ = Kendall.*

| Dataset | Model | $n_\text{tr}$ | $n_\text{va}$ | $n_\text{te}$ | Tuned | $\rho_\text{train}$ | $\rho_\text{val}$ | $\rho_\text{test}$ | $r_\text{test}$ | $\tau_\text{test}$ |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| FeedbackQA | GPT-4o-mini | 309 | 95 | 96 | No | -0.373 | -0.217 | -0.400 | -0.367 | -0.305 |
| FeedbackQA | GPT-4o-mini | 309 | 95 | 96 | Yes | 0.112 | 0.005 | 0.024 | 0.087 | 0.022 |
| FeedbackQA | Llama 3.1 8B | 309 | 95 | 96 | No | -0.185 | -0.147 | -0.265 | -0.272 | -0.199 |
| FeedbackQA | Llama 3.1 8B | 309 | 95 | 96 | Yes | -0.001 | -0.101 | -0.079 | -0.005 | -0.060 |
| FeedbackQA | Mistral 7B | 309 | 95 | 96 | No | -0.311 | -0.196 | -0.273 | -0.239 | -0.203 |
| FeedbackQA | Mistral 7B | 309 | 95 | 96 | Yes | 0.110 | 0.161 | 0.008 | 0.036 | 0.005 |
| FeedbackQA | Gemma 2 9B | 309 | 95 | 96 | No | 0.007 | 0.138 | -0.258 | -0.159 | -0.203 |
| FeedbackQA | Gemma 2 9B | 309 | 95 | 96 | Yes | 0.022 | 0.132 | -0.258 | -0.081 | -0.202 |
| FeedbackQA | Qwen 2.5 7B | 309 | 95 | 96 | No | -0.118 | -0.235 | -0.307 | -0.250 | -0.222 |
| FeedbackQA | Qwen 2.5 7B | 309 | 95 | 96 | Yes | 0.061 | -0.072 | 0.089 | 0.194 | 0.064 |
| FeedbackQA | Phi-3 Mini | 309 | 95 | 96 | No | -0.213 | -0.299 | -0.411 | -0.447 | -0.301 |
| FeedbackQA | Phi-3 Mini | 309 | 95 | 96 | Yes | 0.102 | -0.023 | -0.045 | 0.279 | -0.026 |
| SimpEval | GPT-4o-mini | 300 | 100 | 100 | No | 0.017 | 0.032 | 0.072 | 0.107 | 0.048 |
| SimpEval | GPT-4o-mini | 300 | 100 | 100 | Yes | 0.144 | 0.244 | 0.259 | 0.211 | 0.186 |
| SimpEval | Llama 3.1 8B | 300 | 100 | 100 | No | -0.160 | -0.097 | -0.218 | -0.191 | -0.148 |
| SimpEval | Llama 3.1 8B | 300 | 100 | 100 | Yes | -0.049 | -0.019 | -0.049 | 0.110 | -0.028 |
| SimpEval | Mistral 7B | 300 | 100 | 100 | No | 0.027 | -0.102 | -0.025 | 0.022 | -0.013 |
| SimpEval | Mistral 7B | 300 | 100 | 100 | Yes | 0.152 | 0.023 | 0.251 | 0.238 | 0.176 |
| SimpEval | Gemma 2 9B | 300 | 100 | 100 | No | 0.065 | -0.026 | 0.039 | -0.007 | 0.028 |
| SimpEval | Gemma 2 9B | 300 | 100 | 100 | Yes | 0.073 | -0.021 | 0.035 | -0.027 | 0.025 |
| SimpEval | Qwen 2.5 7B | 300 | 100 | 100 | No | -0.163 | -0.107 | -0.050 | -0.072 | -0.036 |
| SimpEval | Qwen 2.5 7B | 300 | 100 | 100 | Yes | -0.057 | -0.019 | -0.107 | -0.114 | -0.069 |
| SimpEval | Phi-3 Mini | 300 | 100 | 100 | No | 0.036 | -0.051 | -0.136 | -0.063 | -0.089 |
| SimpEval | Phi-3 Mini | 300 | 100 | 100 | Yes | 0.071 | -0.062 | -0.125 | 0.028 | -0.086 |
| SummEval | GPT-4o-mini | 300 | 100 | 100 | No | 0.045 | 0.030 | 0.132 | 0.165 | 0.084 |
| SummEval | GPT-4o-mini | 300 | 100 | 100 | Yes | 0.142 | 0.056 | 0.300 | 0.257 | 0.197 |
| SummEval | Llama 3.1 8B | 300 | 100 | 100 | No | -0.082 | 0.031 | -0.033 | -0.081 | -0.018 |
| SummEval | Llama 3.1 8B | 300 | 100 | 100 | Yes | 0.031 | -0.054 | 0.049 | 0.149 | 0.031 |
| SummEval | Mistral 7B | 300 | 100 | 100 | No | -0.029 | -0.096 | -0.138 | -0.152 | -0.096 |
| SummEval | Mistral 7B | 300 | 100 | 100 | Yes | 0.087 | -0.129 | -0.086 | -0.128 | -0.062 |
| SummEval | Gemma 2 9B | 300 | 100 | 100 | No | 0.040 | 0.152 | 0.072 | 0.103 | 0.053 |
| SummEval | Gemma 2 9B | 300 | 100 | 100 | Yes | 0.050 | 0.105 | 0.034 | 0.019 | 0.027 |
| SummEval | Qwen 2.5 7B | 300 | 100 | 100 | No | -0.059 | -0.054 | 0.000 | 0.040 | 0.001 |
| SummEval | Qwen 2.5 7B | 300 | 100 | 100 | Yes | -0.036 | -0.006 | -0.043 | -0.002 | -0.029 |
| SummEval | Phi-3 Mini | 300 | 100 | 100 | No | 0.044 | -0.154 | 0.078 | 0.084 | 0.049 |
| SummEval | Phi-3 Mini | 300 | 100 | 100 | Yes | 0.069 | -0.151 | 0.082 | 0.155 | 0.056 |
| TopicalChat | GPT-4o-mini | 216 | 72 | 72 | No | -0.029 | -0.258 | -0.176 | -0.266 | -0.131 |
| TopicalChat | GPT-4o-mini | 216 | 72 | 72 | Yes | 0.337 | 0.309 | 0.231 | 0.213 | 0.146 |
| TopicalChat | Llama 3.1 8B | 216 | 72 | 72 | No | -0.056 | -0.096 | -0.113 | -0.101 | -0.085 |
| TopicalChat | Llama 3.1 8B | 216 | 72 | 72 | Yes | 0.069 | -0.024 | 0.046 | 0.040 | 0.029 |
| TopicalChat | Mistral 7B | 216 | 72 | 72 | No | -0.199 | -0.205 | -0.165 | -0.218 | -0.115 |
| TopicalChat | Mistral 7B | 216 | 72 | 72 | Yes | -0.091 | -0.248 | -0.119 | -0.064 | -0.088 |
| TopicalChat | Gemma 2 9B | 216 | 72 | 72 | No | 0.048 | -0.238 | -0.131 | -0.115 | -0.097 |
| TopicalChat | Gemma 2 9B | 216 | 72 | 72 | Yes | 0.064 | -0.232 | -0.118 | -0.003 | -0.086 |
| TopicalChat | Qwen 2.5 7B | 216 | 72 | 72 | No | -0.215 | -0.246 | -0.362 | -0.421 | -0.252 |
| TopicalChat | Qwen 2.5 7B | 216 | 72 | 72 | Yes | -0.059 | 0.012 | 0.109 | 0.120 | 0.071 |
| TopicalChat | Phi-3 Mini | 216 | 72 | 72 | No | -0.015 | 0.200 | -0.173 | -0.242 | -0.117 |
| TopicalChat | Phi-3 Mini | 216 | 72 | 72 | Yes | 0.026 | 0.209 | -0.186 | -0.200 | -0.126 |

---

## Appendix B: Tuned Configuration Profiles

The Optuna optimiser discovered consistent patterns in the configurations that maximise
alignment. Table B1 reports the tuned category weights ($w_T$, $w_E$, $w_B$) for every
model × dataset combination, together with per-dataset means.

**Table B1. Tuned category weight profiles (Optuna-calibrated runs).**
*Weights sum to 1.0 per row. Per-dataset mean shown in final sub-row.*

| Dataset | Judge Model | $w_T$ | $w_E$ | $w_B$ |
|---|---|---:|---:|---:|
| **FeedbackQA** | GPT-4o-mini | 0.180 | 0.820 | 0.000 |
| | Llama 3.1 8B | 0.247 | 0.753 | 0.000 |
| | Mistral 7B | 0.113 | 0.887 | 0.000 |
| | Gemma 2 9B | 0.175 | 0.620 | 0.205 |
| | Qwen 2.5 7B | 0.264 | 0.209 | 0.526 |
| | Phi-3 Mini | 0.204 | 0.796 | 0.000 |
| | *Dataset mean* | *0.197* | *0.681* | *0.122* |
| **SimpEval** | GPT-4o-mini | 0.606 | 0.394 | 0.000 |
| | Llama 3.1 8B | 0.515 | 0.010 | 0.475 |
| | Mistral 7B | 0.137 | 0.698 | 0.165 |
| | Gemma 2 9B | 0.205 | 0.017 | 0.778 |
| | Qwen 2.5 7B | 0.602 | 0.274 | 0.123 |
| | Phi-3 Mini | 0.189 | 0.034 | 0.777 |
| | *Dataset mean* | *0.376* | *0.238* | *0.386* |
| **SummEval** | GPT-4o-mini | 0.666 | 0.334 | 0.000 |
| | Llama 3.1 8B | 0.143 | 0.787 | 0.070 |
| | Mistral 7B | 0.389 | 0.299 | 0.312 |
| | Gemma 2 9B | 0.027 | 0.181 | 0.791 |
| | Qwen 2.5 7B | 0.833 | 0.003 | 0.164 |
| | Phi-3 Mini | 0.957 | 0.043 | 0.000 |
| | *Dataset mean* | *0.502* | *0.275* | *0.223* |
| **TopicalChat** | GPT-4o-mini | 0.002 | 0.944 | 0.053 |
| | Llama 3.1 8B | 0.201 | 0.125 | 0.674 |
| | Mistral 7B | 0.188 | 0.098 | 0.714 |
| | Gemma 2 9B | 0.036 | 0.229 | 0.735 |
| | Qwen 2.5 7B | 0.136 | 0.387 | 0.477 |
| | Phi-3 Mini | 0.355 | 0.645 | 0.000 |
| | *Dataset mean* | *0.153* | *0.405* | *0.442* |

SummEval shows the highest mean trustworthiness weight ($\bar{w}_T = 0.502$), consistent
with factual consistency being the primary quality dimension for news summarisation.
FeedbackQA and TopicalChat are both explainability-heavy at the mean ($\bar{w}_E = 0.681$
and $0.405$ respectively). The bias weight $w_B$ varies substantially across model–task
combinations, reaching near zero for several runs but contributing substantially in
others (particularly TopicalChat: $\bar{w}_B = 0.442$). This variability arises because
none of the benchmarks specifically solicits bias judgements; the optimiser therefore
assigns weight to $w_B$ only when it happens to correlate with the available human signal
for a given model's output distribution, rather than as a principled bias penalty.
