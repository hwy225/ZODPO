# ZODPO
## File layout

```
.
├── train.py                       # entry point
├── trainer.py                     # MeZO / AGZO
├── preference_datasets_hh.py      # data loader
└── config/
    ├── config.yaml                # top-level defaults
    ├── model/
    │   └── qwen306.yaml           # model path + dtype
    ├── trainer/
    │   ├── mezo.yaml              # MeZO hyperparams
    │   └── agzo.yaml              # AGZO hyperparams
    └── loss/
        ├── sft.yaml               # SFT stage config
        └── dpo.yaml               # DPO stage config (requires sft_model_path)
```

## Two-stage training

### Stage 1 — SFT

```bash
# MeZO
python train.py \
    trainer=mezo loss=sft \
    trainer.lr=1e-7 trainer.eps=1e-3 \
    exp_name=sft-mezo-lr1e7-eps1e3 \
    model.name_or_path=Qwen/Qwen3-0.6B \
    total_batches=100 batch_size=16

# AGZO
python train.py \
    trainer=agzo loss=sft \
    trainer.lr=1e-7 trainer.eps=1e-3 \
    exp_name=sft-agzo-lr1e7-eps1e3 \
    model.name_or_path=Qwen/Qwen3-0.6B \
    total_batches=100 batch_size=16
```

SFT checkpoint lands at:  `../runs/<exp_name>/final_model`

### Stage 2 — DPO

```bash
# MeZO-DPO  (load the SFT checkpoint as both the starting policy
#             AND the frozen reference model)
python train.py \
    trainer=mezo loss=dpo \
    trainer.lr=1e-7 trainer.eps=1e-3 \
    exp_name=dpo-agzo-lr1e7-eps1e3 \
    loss.sft_model_path=../runs/<exp_name>/final_model

# AGZO-DPO
python train.py \
    trainer=agzo loss=dpo \
    trainer.lr=1e-7 trainer.eps=1e-3 \
    exp_name=dpo-agzo-lr1e7-eps1e3 \
    loss.sft_model_path=../runs/<exp_name>/final_model
```

## Key config knobs

| Path | Default | Description |
|---|---|---|
| `trainer.lr` | `1e-6` | ZO SGD learning rate |
| `trainer.eps` | `1e-3` | Finite-difference ε |
| `trainer.power_iter_steps` | `5` | (AGZO only) power iteration steps |
| `trainer.rank` | `1` | (AGZO only) activation subspace rank |
| `loss.beta` | `0.1` | (DPO only) KL penalty coefficient |
| `loss.sft_model_path` | `???` | (DPO only) path to the frozen SFT reference |
| `total_batches` | `100` | number of batches to cache and train on |
| `batch_size` | `16` | per-step batch size |
| `max_length` | `512` | max total sequence length |
| `max_prompt_length` | `256` | max prompt length |
