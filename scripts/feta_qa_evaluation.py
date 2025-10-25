# --------------------------- Installing TIGERScore -----------------------------
# Requires cloning repository, removing a private module, and installing from Git
# !rm -rf TIGERScore
# !git clone --depth 1 --no-recurse-submodules https://github.com/TIGER-AI-Lab/TIGERScore.git
# %cd TIGERScore
# !rm -f .gitmodules || true
# !rm -rf hf_space || true
# !ls -al | head -n 30      
# !pip install -e .

# --------------------------- Instantiating TIGERScorer -----------------------------
# Import TIGERScore as the very first module - importing torch / datasets first breaks vLLM multiprocessing
from tigerscore import TIGERScorer

scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-13B", use_vllm=True) # or 7B

# --------------------------- Creating FetaQA Test Vector -----------------------------
# Load the FetaQA dataset
from datasets import load_dataset
metric_instruct_full = load_dataset("TIGER-Lab/MetricInstruct", split="train")
feta_qa_instruction = "Given a table title and some highlighed cells (rows), answer the following question in several sentences with proper reasoning."
metric_instruct_feta_qa = metric_instruct_full.filter(lambda ex: (ex.get("instruction") or "") == feta_qa_instruction)
print(len(metric_instruct_feta_qa)) # 1578

def get_instruction(ex):
    return ex.get("instruction") or ex.get("prompt") or "Answer the question based on the given table."

def get_context(ex):
    return ex.get("input") or ex.get("context") or ex.get("source") or ""

def get_hypo(ex):
    # names vary; these are the usual suspects
    return ex.get("hypo_output") or ex.get("output") or ex.get("candidate") or ""

metric_instruct_feta_qa = (
    metric_instruct_feta_qa.add_column("mi_row_id", list(range(len(metric_instruct_feta_qa))))
)

instructions = [get_instruction(ex) for ex in metric_instruct_feta_qa]
contexts     = [get_context(ex)     for ex in metric_instruct_feta_qa]
hypos        = [get_hypo(ex)        for ex in metric_instruct_feta_qa]
row_ids      = [ex["mi_row_id"]     for ex in metric_instruct_feta_qa]
print("Instructions: " + str(len(instructions)))
print("Contexts: " + str(len(contexts)))
print("Hypos: " + str(len(hypos)))
print("Row IDs: " + str(len(row_ids)))

# --------------------------- Inference with TIGERScore -----------------------------
import json, math, time, pathlib
from tqdm import tqdm

pathlib.Path("tigerscore_outputs").mkdir(exist_ok=True)
outfile = "tigerscore_outputs/fetaqa_metricinstruct_scores.jsonl"

BATCH = 256  # raise on A100/13B; OOM unlikely because VLLM constrains
start = time.time()

# With vLLM - 7 minutes for 13B model. Without VLLM, 4 hours!
with open(outfile, "w", encoding="utf-8") as f:
    for i in tqdm(range(0, len(hypos), BATCH)):
        batch_ins  = instructions[i:i+BATCH]
        batch_ctxs = contexts[i:i+BATCH]
        batch_hyps = hypos[i:i+BATCH]
        batch_ids  = row_ids[i:i+BATCH]

        # scorer.score returns a list of dicts (score, num_errors, errors, raw_output, …)
        results = scorer.score(batch_ins, batch_hyps, batch_ctxs)

        for rid, res in zip(batch_ids, results):
            rec = {"mi_row_id": rid, **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

elapsed = time.time() - start
print(f"Processed {len(hypos)} items in {elapsed:.2f}s → {len(hypos)/elapsed:.2f} ex/s")
print("Saved:", outfile)
