# OneRec RL Training

Reinforcement learning training for OneRec recommendation model based on verl framework.


## Installation

### 1. Configure hostfile (multi-node)

```bash
cat > /etc/mpi/hostfile << EOF
192.168.1.100 slots=8
192.168.1.101 slots=8
192.168.1.102 slots=8
EOF
```

Note: `slots=N` specifies the number of GPUs available on each node.

### 2. Install dependencies

```bash
# Single node
bash deploy_env.sh

# Multi-node
bash deploy_env.sh --all-nodes
```

### 3. Start Ray cluster

```bash
bash init_ray_cluster.sh
```

## Quick Start

### Data Format

We use SFT data from five `*_rec` tasks: `video_rec`, `interactive_rec`, `label_cond_rec`, `ad_rec`, `product_rec`.

See [data/README.md](../data/README.md) for detailed data format specification.

### Start Training

```bash
cd verl_rl

export BASE_MODEL="/path/to/your/model"

bash recipe/onerec/run_grpo.sh 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

## Configuration

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | - | Model path |
| `ROLLOUT_TP_SIZE` | 1 | Tensor parallel size |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 2e-6 | Learning rate |
| `KL_LOSS_COEF` | 0.001 | KL loss coefficient |
| `TEMPERATURE` | 1 | Sampling temperature |

### Rollout

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROLLOUT_N` | 1 | Samples per prompt |
| `STAGE2_BEAM_SIZE` | 32 | Beam search width |
| `RESPONSE_LENGTH` | 2048 | Max response length |
| `STAGE1_MAX_TOKENS` | 1024 | Stage 1 max tokens |
| `STAGE2_NUM_TOKENS` | 3 | Stage 2 tokens |

### Think Mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_THINK` | False | Enable think mode |
| `ENABLE_NONTHINK` | False | Enable non-think mode |
| `USE_FORCE_PREFIX` | False | Force prefix |

## Directory Structure

```
verl_rl/
├── deploy_env.sh          # Environment deployment
├── init_ray.sh            # Single node Ray init
├── init_ray_cluster.sh    # Multi-node Ray cluster
├── requirements.txt       # Dependencies
├── recipe/
│   └── onerec/
│       ├── run_grpo.sh    # Training script
│       └── onerec_recipe.py
└── verl/                  # verl core code
```
