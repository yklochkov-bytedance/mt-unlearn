# Unlearning with mean teacher

Supplementary code for the paper "A mean teacher algorithm for unlearning of language models"

# Reproducing experiments

## Step 1: preparation

Follow instructions in ```prepare/README.md```

## Step 2: run unlearning scripts

Mean teacher example:
```bash
export CONFIG_PATH="example_configs/mt_nlul_qkl.yaml"
python3 run_mt.py
```

NPO example:
```bash
export CONFIG_PATH="example_configs/adamw_npo_kl.yaml"
python3 run_mt.py
```

# Details of experiments in the paper

We provide the logs for the experiments in the paper in ```logs``` folder.

# Environment

The following packages must be installed:

```
torch==2.1.2
vllm==0.3.0
transformers
datasets
tqdm
scipy
scikit-learn
rouge_score
```
**WARNING!** Our code does not work with ```torch>2.1.2``` since we rely on the beta-version of ```torch.distributed``` in ```tp_utils```
