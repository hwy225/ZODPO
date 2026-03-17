# ZODPO
## File layout

```
.
├── train.py                       # entry point
├── trainer.py                     # MeZO / AGZO / AGZOPlain
├── preference_datasets_hh.py      # data loader
└── config/
    ├── config.yaml                # top-level defaults
    ├── model/
    │   └── qwen306.yaml           # model path + dtype
    ├── trainer/
    │   ├── mezo.yaml              # MeZO hyperparams
    │   ├── agzo_plain.yaml        # Plain AGZO hyperparams
    │   └── agzo.yaml              # AGZO hyperparams
    └── loss/
        ├── sft.yaml               # SFT stage config (TRL)
        └── dpo.yaml               # DPO stage config (requires sft_model_path)
```

## Two-stage training

### Stage 1 — SFT

```bash
python train.py loss=sft loss.lr=5e-5 exp_name=sft-lr5e-5
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

# Plain-AGZO-DPO
python train.py \
    trainer=agzo_plain loss=dpo \
    trainer.lr=1e-7 trainer.eps=1e-3 \
    exp_name=dpo-agzo-plain-lr1e7-eps1e3 \
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
| `loss.lr` | `5e-5` | (SFT) SGD learning rate |
| `loss.lr_scheduler_type` | `cosine` | (SFT) SGD LR schedule |
| `loss.warmup_ratio` | `0.03` | (SFT) SGD warmup ratio |
| `loss.num_train_epochs` | `1` | (SFT) SGD training epochs |
| `trainer.lr` | `1e-6` | (DPO/ZO) ZO learning rate |
| `trainer.eps` | `1e-3` | (DPO/ZO) ZO finite-difference ε |
| `trainer.power_iter_steps` | `5` | (AGZO) Power iteration steps |
| `trainer.rank` | `1` | (AGZO) Subspace rank |
| `loss.beta` | `0.1` | (DPO) KL penalty coefficient |
| `loss.sft_model_path` | `???` | (DPO) Path to frozen SFT reference |
| `total_batches` | `100` | Total batches (SFT max_steps / DPO cache) |
| `batch_size` | `16` | Per-step batch size |
| `max_length` | `512` | Max total sequence length |
| `max_prompt_length` | `256` | Max prompt length |
