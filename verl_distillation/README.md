## Overview

This repository is built on top of the open-source [**verl**](https://github.com/volcengine/verl) (HybridFlow RLHF/RL training framework) and adds support for **on-policy distillation**.
It is designed for scenarios where the **teacher and student use different vocabularies**, e.g., distilling from `Qwen3` (teacher) to a recommendation-pretrained model (student) that contains **extended itemic tokens**, while improving and preserving general-purpose capabilities.

> **Note**: This repository is forked from [verl](https://github.com/volcengine/verl) at commit [`703a078`](https://github.com/volcengine/verl/commit/703a07856fe2544833dfce51136f386654574b30) and extended with on-policy distillation capabilities.

The high-level idea is briefly described in the OpenOneRec technical report, Section **5.2 On-policy Distillation for General Capability**: [OneRecBench.pdf](OneRecBench.pdf).

## Key Features

- **On-policy distillation entrypoint**: `recipe/onpolicy_distill/main_onpolicy_distill.py`
- **Distillation trainer**: `recipe/onpolicy_distill/onpolicy_distill_trainer.py`
- **Teacher/Student vocabulary mismatch support**
  - Generates `distill_special_token_mask` during rollout
  - Replaces/masks extended-vocab tokens during log-probability computation to improve training stability
- **OneRec dataset adapter (parquet → chat)**: `verl/utils/dataset/onerec_dataset.py`
  - Optionally appends `/think` or `/no_think` to the user prompt (force/auto modes)
- **Algorithm and metrics extensions**
  - `AdvantageEstimator.ON_POLICY_DISTILL`
  - `compute_on_policy_distill_data_metrics(...)`

## Quick Start

### Installation
```bash
# Configure hostfile (multi-node)
cat > /etc/mpi/hostfile << EOF
192.168.1.100
192.168.1.101
192.168.1.102
EOF

# Install dependencies
# For Single node
bash deploy_env.sh
# For Multi-node
bash deploy_env.sh --all-nodes

# Start Ray cluster
bash init_ray_cluster.sh
```


### Required environment variables

```bash
# Required: model and data paths
export BASE_MODEL=/path/to/student_model
export TEACHER_MODEL=/path/to/teacher_model   # e.g. Qwen3-1.7B

# Optional: extended-vocabulary distillation settings (defaults in the script)
export EXTEND_VOCAB_START_TOKEN=151669         # token_id >= this value is treated as an "extended vocab token"
export MASK_RESPONSE_IF_HAVE_EXTEND_TOKEN=False  # mask the whole response if any extended token appears

# Optional: advantage clipping bounds for distillation (defaults in the script)
export DISTILL_ADV_MAX=5.0    # upper bound
export DISTILL_ADV_MIN=-30.0  # lower bound
```

**`EXTEND_VOCAB_START_TOKEN`**
is used for teacher/student vocabulary mismatch. If the student model introduces additional tokens on top of the base vocabulary (e.g., item tokens for recommendation), set this threshold to the first extended token id.
During rollout, the framework produces `distill_special_token_mask`; during log-probability computation, extended-vocab tokens are replaced/masked to maintain stability.

**`DISTILL_ADV_MAX / DISTILL_ADV_MIN`**
clip the distillation advantage to avoid extreme values when the teacher and student distributions differ substantially. The distillation signal is token-level reverse KL:
$$A = -(\log p_{\text{student}} - \log p_{\text{teacher}})$$

### Launch training

The training entry script is located at `recipe/onpolicy_distill/run_qwen3_distill.sh`.

```bash
bash recipe/onpolicy_distill/run_qwen3_distill.sh /etc/mpi/hostfile
```

Notes:
- The script defaults to **console-only logging** (`trainer.logger=[console]`). To use W&B, export `WANDB_API_KEY` and override `trainer.logger=[console,wandb]` in the script/CLI.
- Hydra config entrypoint: `recipe/onpolicy_distill/config/onpolicy_distill_trainer.yaml` (reuses the base config from [verl](https://github.com/volcengine/verl)).

## Data Format (parquet)

`OneRecDataset` reads the `messages` field from parquet (either a list, or a string-serialized list) and constructs:
- `prompt`: all messages except the last one
- `ground_truth`: the content of the last message (used for reward payload / analysis)

It is recommended to keep a `source` or `data_source` field for per-task statistics.

## Key Implementation Details (for reproducibility)

- **Distillation signal (reverse KL)**
  - Implemented in `verl/trainer/ppo/core_algos.py` as:
    \(A = -(\log p_{\text{student}} - \log p_{\text{teacher}})\)
  - Enabled via the `compute_advantage(...)` branch in `verl/trainer/ppo/ray_trainer.py`, with support for `distill_adv_max_clip / distill_adv_min_clip`.

- **Extended vocabulary handling**
  - `extend_vocab_start_token`: tokens with id \(\ge\) this threshold are treated as "extended vocab tokens"
  - `ToolAgentLoop` emits `distill_special_token_mask` (optionally truncating/masking the response)
  - `dp_actor.compute_log_prob(..., mask_special_token=True)` replaces/masks extended-vocab tokens and overwrites the corresponding log-prob entries (via `ref_log_prob_replace_val`)

---

## 🙏 Acknowledgements

This repository is built upon and extended from the open-source [**verl**](https://github.com/volcengine/verl) project. We sincerely thank the verl team for their excellent work on the HybridFlow RLHF/RL training framework, which provides the solid foundation for our on-policy distillation implementation.
