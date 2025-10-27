# FaceID Project

Minimal training/evaluation pipeline for face verification using a ResNet-50 backbone.
Supports two training objectives:
- pair: Binary-classification on image pairs (BCE on cosine similarity)
- arcface: Classification with ArcMarginProduct (ArcFace)

Recommended: arcface + pretrained backbone for stable and strong performance.

## Repo layout

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

## Quick start (ArcFace - khuyến nghị)

Pretrained ResNet-50, đóng băng 2 epoch, unfreeze với LR nhỏ cho backbone, cosine + warmup, kẹp gradient:

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

## Results summary

- Pair-BCE (from scratch):
  - AUC ≈ 0.52–0.57; TAR@1e-2 ≈ 0.01–0.02; TAR@1e-3 ≈ 0.000–0.006 (gần ngẫu nhiên).
- ArcFace (pretrained, freeze=2, cosine+warmup):
  - Best around epoch 5:
    - acc=0.9489, auc=0.9815, TAR@1e-2=0.8438, TAR@1e-3=0.5397, th≈0.95
  - Dấu hiệu overfit nhẹ sau đó; có thể early-stop theo AUC/TAR.

Checkpoints: `models/backbone_best.pt` (theo acc). You may extend to save by AUC/TAR.

## Metrics

- acc: accuracy trên cặp mẫu cân bằng pos/neg (balanced pairs).
- th: ngưỡng cosine tối ưu theo acc ([-1, 1]).
- AUC: ROC-AUC trên cặp mẫu.
- TAR@FAR: True Accept Rate tại FAR mục tiêu (ví dụ 1e-2, 1e-3).

## Pair mode (tham khảo)

Chạy từ đầu (không khuyến nghị cho chất lượng):
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
Gợi ý nếu vẫn dùng pair mode: thêm `--pretrained --freeze_backbone_epochs 2` hoặc chuyển sang contrastive/triplet loss.

## Notes

- Đã bật mixed precision + TF32 (nếu GPU hỗ trợ), cuDNN benchmark, channels_last.
- Tăng tốc / ổn định: `--clip_grad_norm`, chỉnh `--arcface_m`, `--backbone_lr_scale`, `--warmup_epochs`.
- Phần lớn file sinh ra (models/, data/, logs/, results/) đã được bỏ qua bởi .gitignore.
