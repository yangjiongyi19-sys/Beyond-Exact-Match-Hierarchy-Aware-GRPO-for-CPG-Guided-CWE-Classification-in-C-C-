# Hierarchy-Aware CPG Exploration for CWE Classification

This repository contains a research implementation for C/C++ vulnerability classification. The model is first supervised fine-tuned (SFT) to learn the output schema and basic classification, then optimized with standard Group Relative Policy Optimization (GRPO) to learn whether to query the code graph, which query to make, and when to stop.

The central research question is: **Under the same tool budget, does a CWE-hierarchy reward improve classification compared with exact-match reward?** See the [detailed experiment plan](EXPERIMENT_PLAN.md) for hypotheses, comparisons, statistical analysis, and projected results.

## Method

1. **SFT initialization:** Train on function-level examples. The target is JSON containing `verdict`, `cwe_id`, and `evidence_node_ids`; labels are not included in the input prompt.
2. **Multi-turn environment:** Provide three read-only CPG operations: `get_callers`, `get_callees`, and `dataflow_slice`. Each sample allows at most three queries. The model ends the interaction by returning its final prediction.
3. **Standard GRPO:** Sample multiple complete trajectories for each function and optimize them using their terminal rewards and within-group comparisons. No turn-level reward shaping or custom VERL advantage calculation is used.
4. **Reward comparison:** The proposed method uses a normalized ALPHA score over the CWE hierarchy. The control replaces this term with exact CWE matching. Both methods share the evidence-validity and query-efficiency terms. There is no separate binary-correctness reward term.

The terminal reward is:

```text
R = 0.80 × R_task + 0.10 × R_evidence + 0.10 × R_efficiency
```

For vulnerable samples, `R_task` is the normalized ALPHA score; in the control, it is exact CWE-match accuracy. For safe samples, predicting safe scores 1 and predicting vulnerable scores 0. Thus safe/vulnerable decisions are part of the task score, without an additional `R_binary` term. `R_evidence` checks whether cited node IDs appeared in the initial observation or tool results. Safe predictions do not need to cite vulnerability nodes. `R_efficiency = 1 − query_count/3`. Binary F1 is used for evaluation only. Rewards are computed deterministically, without an LLM judge, and are not shown to the model.

The harness provides the environment for multi-turn RL: each query returns new code observations with stable node IDs. Repeatedly generating text from the same input is not treated as active exploration. The harness is read-only and does not execute model-generated code or queries.

## Data and Evaluation

- **PrimeVul:** Primary training, validation, and test dataset, using the official split.
- **BigVul:** External evaluation. CWE metrics are reported only on a manually audited subset; unaudited CWE labels are not treated as reliable reward signals or evidence for conclusions.
- **Deduplication:** Use normalized function-code hashes and, when available, CVE, commit, and project metadata to identify cross-split duplicates. Prefer PrimeVul official test samples when resolving overlap.
- **Taxonomy:** Freeze the MITRE CWE 4.20 snapshot for training rewards and offline evaluation.

Report CWE Macro-F1, normalized ALPHA, binary F1, mean query count, and evidence-node validity. The main comparisons are ordinary SFT, tool-enabled GRPO with exact CWE reward, and tool-enabled GRPO with hierarchical ALPHA reward. Use three random seeds and paired bootstrap confidence intervals. Select checkpoints using validation data only.

The central claim is falsifiable: if hierarchical-reward GRPO does not outperform exact-match GRPO under a fixed query budget, or if CWE gains come with a substantial decline in binary F1, the results do not support the proposed claim.

## Quick Start

Commands below assume a Linux training machine with Joern, VERL, and SGLang installed. Pin dependency versions, especially the VERL AgentLoop API.

### 1. Prepare the datasets and CWE taxonomy

```bash
pip install -e '.[data,train]'
vulnrl-prepare \
  --primevul ./data/primevul.jsonl \
  --bigvul ./data/bigvul.csv \
  --out-dir ./artifacts/data \
  --no-bigvul-augmentation

vulnrl-taxonomy \
  --xml ./data/cwec_v4.20.xml \
  --out ./artifacts/cwe_4_20.json
```

BigVul can first be used for external evaluation. Do not use its CWE labels for RL training unless an audit mapping is available. `bigvul_eval.jsonl` excludes functions duplicated in PrimeVul. To provide an audit table with `sample_id,cwe_id`, add `--bigvul-cwe-audit ./data/bigvul_cwe_audit.csv` to the preparation command.

Export the training and validation files for GRPO:

```bash
vulnrl-export-verl --input ./artifacts/data/primevul_train.jsonl --out ./artifacts/train.parquet
vulnrl-export-verl --input ./artifacts/data/primevul_val.jsonl --out ./artifacts/valid.parquet
```

### 2. Prepare read-only CPG snapshots

Generate a Joern CPG offline for each fixed repository revision, export it as GraphML, and convert it. A CPG can be reused by multiple functions from the same repository snapshot.

```bash
vulnrl-joern-graphml \
  --graphml ./joern/repo/export/cpg.graphml \
  --project-id repo@commit \
  --out ./artifacts/cpg/repo.json

vulnrl-build-cpg-index \
  --dataset ./artifacts/data/primevul_train.jsonl \
  --dataset ./artifacts/data/primevul_val.jsonl \
  --dataset ./artifacts/data/primevul_test.jsonl \
  --dataset ./artifacts/data/bigvul_eval.jsonl \
  --project-map ./project_map.json \
  --snapshot-dir ./artifacts/cpg \
  --out ./artifacts/cpg/index.json
```

Check unmatched and ambiguous functions after indexing. The harness exposes only call-graph and data-flow operations and returns a bounded number of code nodes.

### 3. SFT initialization

```bash
vulnrl-build-sft-examples \
  --input ./artifacts/data/primevul_train.jsonl \
  --out ./artifacts/sft.jsonl

torchrun --standalone --nproc_per_node=4 -m vulnrl.train_sft \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --traces ./artifacts/sft.jsonl \
  --out /mnt/checkpoints/cwe-classifier-sft
```

### 4. Run tool-enabled GRPO

```bash
export TRAIN_PARQUET=/mnt/data/train.parquet
export VAL_PARQUET=/mnt/data/primevul_valid.parquet
export TAXONOMY_JSON=/mnt/data/cwe_4_20.json
export CPG_SNAPSHOT_DIR=/mnt/data/cpg
export MODEL_PATH=/mnt/checkpoints/cwe-classifier-sft
export CHECKPOINT_DIR=/mnt/checkpoints/cwe-grpo-alpha
export HIERARCHY_REWARD_MODE=alpha
bash scripts/run_grpo.sh
```

To run the exact-match control, keep the data, initialization, and hyperparameters the same, and use a separate output directory:

```bash
export CHECKPOINT_DIR=/mnt/checkpoints/cwe-grpo-exact
export HIERARCHY_REWARD_MODE=exact
bash scripts/run_grpo.sh
```

See [configs/experiment.yaml](configs/experiment.yaml) for the configuration. The main settings are Qwen2.5-Coder-7B, 4×H100 80GB, eight rollouts per prompt, learning rate `3e-6`, and up to three CPG queries. The training script uses one fixed configuration and does not apply a multi-stage interaction-length curriculum.

### 5. Evaluate

Evaluate PrimeVul as usual. For BigVul, enable the audit filter so CWE metrics use only manually audited labels:

```bash
vulnrl-evaluate \
  --labels ./artifacts/data/bigvul_eval.jsonl \
  --predictions ./artifacts/bigvul_predictions.jsonl \
  --taxonomy ./artifacts/cwe_4_20.json \
  --require-audited-cwe \
  --out ./artifacts/bigvul_metrics.json
```

With this option, binary metrics still use all BigVul samples, while CWE metrics use only rows marked `cwe_label_status=audited`.

## Relation to Prior Work

Li et al. study CPG tools and multi-turn RL for vulnerability detection; Adnan et al. study hierarchy-aware reward for single-turn CWE prediction. This project does not claim that tool use or hierarchical reward alone is novel. It tests the narrower question of whether a complete CWE-hierarchy score, used as a terminal reward in multi-turn CPG exploration, improves CWE classification and query efficiency.

- [Li et al., *Graph Is the Verifier*](https://arxiv.org/abs/2607.26656)
- [Adnan et al., *From Evaluation to Optimisation*](https://arxiv.org/abs/2607.21069)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [VERL AgentLoop documentation](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
