# FaceID Project

Minimal training and evaluation pipeline for face verification using a ResNet-50 backbone.

Two training objectives are supported:
- pair: Binary classification on image pairs (BCE on cosine similarity)
- arcface: Classification with ArcMarginProduct (ArcFace)

Recommendation: use ArcFace with a pretrained backbone for stable and strong performance.

## Repository layout

```
src/
  dataset.py   # datasets and transforms
  model.py     # ResNet-50 backbone + ArcMarginProduct
  train.py     # training + evaluation (AUC, TAR@FAR)
  utils.py     # seeds, metrics, helpers

models/        # checkpoints (ignored by git)
data/          # ImageFolder layout: train/ and val/
```

Expected data structure (ImageFolder):
```
<split>/
  person_0001/*.jpg
  person_0002/*.jpg
  ...
```

## Quick start (ArcFace - recommended)

Pretrained ResNet-50, freeze 2 epochs, unfreeze with a smaller LR for the backbone, cosine schedule with warmup, gradient clipping:

```bash
python src/train.py \
  --train_mode arcface \
  --pretrained \
  --freeze_backbone_epochs 2 \
  --epochs 30 \
  --workers 32 \
  --eval_pairs 100000 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --backbone_lr_scale 0.1 \
  --progress_step_pct 1 \
  --clip_grad_norm 1.0 \
  --arcface_s 64 \
  --arcface_m 0.5 \
  --train /home/hoc/MyProjects/faceid-project/data/train \
  --val   /home/hoc/MyProjects/faceid-project/data/val \
  --ckpt_dir /home/hoc/MyProjects/faceid-project/models
```

Defaults: batch_size=256, img_size=224.

### Environment setup (quick suggestion)

```bash
# Optional: create a virtual environment
python -m venv venv-pytorch
source venv-pytorch/bin/activate

# Install PyTorch + torchvision (example for CUDA 12.x)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install numpy pillow scikit-learn
```

If your GPU/driver differs, see the official installation guide at pytorch.org.

## Results summary

- Pair-BCE (from scratch):
  - AUC ≈ 0.52–0.57; TAR@1e-2 ≈ 0.01–0.02; TAR@1e-3 ≈ 0.000–0.006 (near-random).
- ArcFace (pretrained, freeze=2, cosine+warmup):
  - Best around epoch 5:
    - acc=0.9489, auc=0.9815, TAR@1e-2=0.8438, TAR@1e-3=0.5397, th≈0.95
  - Slight overfitting/oscillation afterwards; consider early stopping by AUC/TAR.

Checkpoints: `models/backbone_best.pt` (by default, selected by accuracy). You can select a different metric with `--best_metric`:

```bash
# Save best checkpoint by AUC
python src/train.py --train_mode arcface --pretrained --best_metric auc ...

# Save by TAR@FAR=1e-3
python src/train.py --train_mode arcface --pretrained --best_metric tar_1e-3 ...
```

## Metrics

- acc: accuracy on balanced positive/negative pairs.
- th: cosine threshold maximizing accuracy (range [-1, 1]).
- AUC: ROC-AUC over sampled verification pairs.
- TAR@FAR: True Accept Rate at a given False Accept Rate (e.g., 1e-2, 1e-3).

## Pair mode (reference)

From-scratch training (not recommended for quality):
```bash
python src/train.py \
  --epochs 40 \
  --workers 32 \
  --pairs_per_epoch 60000 \
  --eval_pairs 100000 \
  --scale_logits 64 \
  --scheduler cosine \
  --warmup_epochs 3 \
  --progress_step_pct 1 \
  --train /home/hoc/MyProjects/faceid-project/data/train \
  --val   /home/hoc/MyProjects/faceid-project/data/val \
  --ckpt_dir /home/hoc/MyProjects/faceid-project/models
```
If you stay with pair mode, prefer `--pretrained --freeze_backbone_epochs 2`, or switch to contrastive/triplet losses for stronger supervision.

## Notes

- Mixed precision and TF32 (if supported) are enabled, as well as cuDNN benchmark and channels_last.
- Useful knobs: `--clip_grad_norm`, tune `--arcface_m`, `--backbone_lr_scale`, `--warmup_epochs`.
- Most generated artifacts (models/, data/, logs/, results/) are ignored by .gitignore.
