"""
Command:
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
  --val /home/hoc/MyProjects/faceid-project/data/val \
  --ckpt_dir /home/hoc/MyProjects/faceid-project/models

  Result:
  [train 05/30] 100%
  Epoch 05/30 train_loss=18.3621 time=1186.2s
  [val   05/30] 100%
  val_pairs=100000 (pos=50000, neg=50000)  acc=0.9489  th=0.950  auc=0.9815  TAR@1e-2=0.8438  TAR@1e-3=0.5397
"""
import os
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

# Project-specific modules (already in your repo)
from dataset import ImageFolderPairs, build_transforms
from model import BackboneResNet50, ArcMarginProduct
from utils import set_seed, compute_verif_metrics

def main(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Modern TF32 / precision knobs (PyTorch 2.9+ nightly)
    try:
        # TF32 for cuDNN convs (faster on Ampere+)
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        # TF32 for CUDA matmul; use 'ieee' for strict FP32 if needed
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
    except Exception:
        pass

    # Let cuDNN autotune best conv algorithms for current shapes/batch
    torch.backends.cudnn.benchmark = True

    # ----- Data -----    
    _tfms = build_transforms(args.img_size, use_imagenet_norm=args.pretrained)
    if isinstance(_tfms, (list, tuple)):
        if len(_tfms) >= 2:
            T_train, T_eval = _tfms[0], _tfms[1]
        else:
            T_train = T_eval = _tfms[0]
    else:
        T_train = T_eval = _tfms

    # Build datasets/loaders depending on training mode
    if args.train_mode == 'pair':
        train_ds = ImageFolderPairs(
            root=args.train, transform=T_train,
            pairs_per_epoch=args.pairs_per_epoch, pos_ratio=args.pos_ratio
        )
    else:
        # classification mode: plain ImageFolder
        train_ds = datasets.ImageFolder(args.train, transform=T_train)

    val_ds   = datasets.ImageFolder(args.val, transform=T_eval)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=6
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(128, args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=6
    )

    # ----- Model -----
    backbone = BackboneResNet50(embed_dim=args.embed_dim, pretrained=args.pretrained).to(device).to(memory_format=torch.channels_last)
    arc_head = None
    if args.train_mode == 'arcface':
        num_classes = len(getattr(train_ds, 'classes', []))
        if num_classes is None or num_classes == 0:
            raise RuntimeError("No classes found in training dataset for ArcFace mode.")
        arc_head = ArcMarginProduct(
            in_features=args.embed_dim,
            out_features=num_classes,
            s=float(args.arcface_s),
            m=float(args.arcface_m)
        ).to(device)

    # Optional: freeze backbone for first N epochs, only train the projection head
    if args.freeze_backbone_epochs > 0:
        for p in backbone.backbone.parameters():
            p.requires_grad = False

    # Build optimizer param groups for better LR control
    head_params = list(backbone.head.parameters())
    if arc_head is not None:
        head_params += list(arc_head.parameters())
    bb_params = list(backbone.backbone.parameters())

    # Optionally freeze backbone initially
    if args.freeze_backbone_epochs > 0:
        for p in bb_params:
            p.requires_grad = False

    param_groups = [
        {"params": head_params, "lr": args.lr, "weight_decay": 1e-4, "name": "head"}
    ]
    # If not freezing, include backbone immediately with scaled LR
    if args.freeze_backbone_epochs == 0:
        param_groups.append({
            "params": bb_params,
            "lr": args.lr * args.backbone_lr_scale,
            "weight_decay": 1e-4,
            "name": "backbone"
        })

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Track best by selected metric
    best_metric_val = float('-inf')
    no_improve_epochs = 0
    best_acc = 0.0  # kept for backward-compat logging if needed
    scale_logits = float(args.scale_logits)  # scale for cosine logits

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        t0 = time.time()
        running = 0.0

        # Refresh pairs each epoch for pair-training to improve sampling diversity
        if args.train_mode == 'pair' and hasattr(train_ds, '_make_pairs'):
            try:
                train_ds.pairs = train_ds._make_pairs()
            except Exception:
                pass

        # Unfreeze backbone after the configured number of warmup epochs
        if args.freeze_backbone_epochs > 0 and epoch == (args.freeze_backbone_epochs + 1):
            for p in backbone.backbone.parameters():
                p.requires_grad = True
            # Add backbone params as a new param group without resetting optimizer state for the head
            backbone_params = [p for p in backbone.backbone.parameters() if p.requires_grad]
            if len(backbone_params) > 0:
                optimizer.add_param_group({
                    "params": backbone_params,
                    "lr": args.lr * args.backbone_lr_scale,
                    "weight_decay": 1e-4,
                    "name": "backbone",
                })

        # ----- LR schedule (optional warmup + cosine) -----
        def lr_multiplier(e):
            if args.scheduler == 'none':
                return 1.0
            # 1-based epoch indexing here
            warm = max(0, int(args.warmup_epochs))
            total = max(1, int(args.epochs))
            ee = min(e, total)
            if warm > 0 and ee <= warm:
                return float(ee) / float(warm)
            # cosine decay from 1.0 to 0.0 over remaining epochs
            t = (ee - warm) / max(1, (total - warm))
            return 0.5 * (1.0 + math.cos(math.pi * t))

        mult = lr_multiplier(epoch)
        for g in optimizer.param_groups:
            g_name = g.get('name', '')
            if g_name == 'head':
                g['lr'] = args.lr * mult
            elif g_name == 'backbone':
                g['lr'] = args.lr * args.backbone_lr_scale * mult

        # Train progress tracking setup
        step_frac = (args.progress_step_pct / 100.0) if args.progress_step_pct and args.progress_step_pct > 0 else None
        next_mark = step_frac if step_frac else None
        total_batches = len(train_loader)
        train_last_len = 0
        train_last_ts = 0.0  # rate-limit printing (max ~10 Hz)

        if args.train_mode == 'pair':
            for b_idx, (x1, x2, y) in enumerate(train_loader):
                x1 = x1.to(device, non_blocking=True, memory_format=torch.channels_last)
                x2 = x2.to(device, non_blocking=True, memory_format=torch.channels_last)
                y  = y.to(device, non_blocking=True).float().view(-1)  # [B]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    e1 = backbone(x1)  # [B, D], L2-normalized by model
                    e2 = backbone(x2)  # [B, D]
                    cos = (e1 * e2).sum(dim=1)           # [-1, 1], shape [B]
                    logits = cos * scale_logits          # logits for BCEWithLogits
                    loss = F.binary_cross_entropy_with_logits(logits, y)

                scaler.scale(loss).backward()
                # Optional gradient clipping (after unscale, before step)
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad],
                            max_norm=float(args.clip_grad_norm)
                        )
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()

                # Print train progress if enabled (single updating line)
                if next_mark is not None and total_batches > 0:
                    prog = float(b_idx + 1) / float(total_batches)
                    while prog + 1e-12 >= next_mark and next_mark <= 1.0 + 1e-12:
                        pct = min(next_mark * 100.0, 100.0)
                        pct_str = f"{pct:.2f}".rstrip('0').rstrip('.')
                        # Rate-limit to <= 10 prints/sec, but always print 100%
                        now = time.time()
                        if (now - train_last_ts) >= 0.1 or pct >= 100.0:
                            msg = f"[train {epoch:02d}/{args.epochs}] {pct_str}%"
                            padding = " " * max(0, train_last_len - len(msg))
                            print("\r" + msg + padding, end='', flush=True)
                            train_last_len = len(msg)
                            train_last_ts = now
                        next_mark += step_frac
        else:
            # ArcFace classification training
            ce_loss = nn.CrossEntropyLoss()
            for b_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    emb = backbone(xb)              # [B, D], L2-normalized
                    logits = arc_head(emb, yb)      # ArcMargin output
                    loss = ce_loss(logits, yb)

                scaler.scale(loss).backward()
                # Optional gradient clipping (after unscale, before step)
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g['params'] if p.requires_grad],
                            max_norm=float(args.clip_grad_norm)
                        )
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()

                running += loss.item()

                # Print train progress if enabled (single updating line)
                if next_mark is not None and total_batches > 0:
                    prog = float(b_idx + 1) / float(total_batches)
                    while prog + 1e-12 >= next_mark and next_mark <= 1.0 + 1e-12:
                        pct = min(next_mark * 100.0, 100.0)
                        pct_str = f"{pct:.2f}".rstrip('0').rstrip('.')
                        now = time.time()
                        if (now - train_last_ts) >= 0.1 or pct >= 100.0:
                            msg = f"[train {epoch:02d}/{args.epochs}] {pct_str}%"
                            padding = " " * max(0, train_last_len - len(msg))
                            print("\r" + msg + padding, end='', flush=True)
                            train_last_len = len(msg)
                            train_last_ts = now
                        next_mark += step_frac

            # Print train progress if enabled (single updating line)
            if next_mark is not None and total_batches > 0:
                prog = float(b_idx + 1) / float(total_batches)
                while prog + 1e-12 >= next_mark and next_mark <= 1.0 + 1e-12:
                    pct = min(next_mark * 100.0, 100.0)
                    pct_str = f"{pct:.2f}".rstrip('0').rstrip('.')
                    # Rate-limit to <= 10 prints/sec, but always print 100%
                    now = time.time()
                    if (now - train_last_ts) >= 0.1 or pct >= 100.0:
                        msg = f"[train {epoch:02d}/{args.epochs}] {pct_str}%"
                        padding = " " * max(0, train_last_len - len(msg))
                        print("\r" + msg + padding, end='', flush=True)
                        train_last_len = len(msg)
                        train_last_ts = now
                    next_mark += step_frac

        # Finish train progress line if it was printed
        if next_mark is not None:
            print()
        dt = time.time() - t0
        avg_loss = running / max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={avg_loss:.4f} time={dt:.1f}s")

        # ----- Validation -----
        backbone.eval()
        with torch.no_grad():
            all_embs = []
            all_labels = []

            # Val progress tracking setup
            v_step_frac = (args.progress_step_pct / 100.0) if args.progress_step_pct and args.progress_step_pct > 0 else None
            v_next_mark = v_step_frac if v_step_frac else None
            v_total_batches = len(val_loader)
            v_last_len = 0
            v_last_ts = 0.0  # rate-limit printing (max ~10 Hz)

            for v_idx, (xb, yb) in enumerate(val_loader):
                xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    eb = backbone(xb).float()  # move to fp32 for metrics
                all_embs.append(eb.cpu())
                all_labels.append(yb.cpu())

                # Print val progress if enabled (single updating line)
                if v_next_mark is not None and v_total_batches > 0:
                    v_prog = float(v_idx + 1) / float(v_total_batches)
                    while v_prog + 1e-12 >= v_next_mark and v_next_mark <= 1.0 + 1e-12:
                        v_pct = min(v_next_mark * 100.0, 100.0)
                        v_pct_str = f"{v_pct:.2f}".rstrip('0').rstrip('.')
                        # Rate-limit to <= 10 prints/sec, but always print 100%
                        v_now = time.time()
                        if (v_now - v_last_ts) >= 0.1 or v_pct >= 100.0:
                            v_msg = f"[val   {epoch:02d}/{args.epochs}] {v_pct_str}%"
                            v_padding = " " * max(0, v_last_len - len(v_msg))
                            print("\r" + v_msg + v_padding, end='', flush=True)
                            v_last_len = len(v_msg)
                            v_last_ts = v_now
                        v_next_mark += v_step_frac

            # Finish val progress line if it was printed
            if v_next_mark is not None:
                print()

            E = torch.cat(all_embs, dim=0)
            L = torch.cat(all_labels, dim=0)
            metrics = compute_verif_metrics(E, L, max_pairs=args.eval_pairs)

        tars = metrics.get('tar_at_far', {})
        # Build a compact string like: TAR@1e-2=0.8123 TAR@1e-3=0.7011
        tar_strs = []
        for k in sorted(tars.keys()):
            v = tars[k]
            tar_val = v.get('tar', float('nan')) if isinstance(v, dict) else float('nan')
            tar_strs.append(f"TAR@{k}={tar_val:.4f}")
        tar_block = '  ' + '  '.join(tar_strs) if tar_strs else ''

        print(
            "  val_pairs={pairs} (pos={pos}, neg={neg})  acc={acc:.4f}  th={th:.3f}  auc={auc:.4f}{extra}".format(
                pairs=metrics.get('pairs', 0),
                pos=metrics.get('pos_pairs', 0),
                neg=metrics.get('neg_pairs', 0),
                acc=metrics.get('best_acc', 0.0),
                th=metrics.get('best_th', 0.0),
                auc=metrics.get('auc', float('nan')) if metrics.get('auc', None) is not None else float('nan'),
                extra=tar_block,
            )
        )
        # ----- Select best checkpoint by chosen metric -----
        def metric_value(mdict, key: str):
            if key == 'acc':
                return float(mdict.get('best_acc', float('nan')))
            if key == 'auc':
                return float(mdict.get('auc', float('nan')))
            if key == 'tar_1e-2':
                v = mdict.get('tar_at_far', {}).get('1e-2', {})
                return float(v.get('tar', float('nan')))
            if key == 'tar_1e-3':
                v = mdict.get('tar_at_far', {}).get('1e-3', {})
                return float(v.get('tar', float('nan')))
            return float('nan')

        curr = metric_value(metrics, args.best_metric)
        # Treat NaN as -inf for comparison
        if curr != curr:  # NaN check
            curr_cmp = float('-inf')
        else:
            curr_cmp = curr
        improved = False
        # Require improvement greater than min_delta (if configured)
        if curr_cmp > (best_metric_val + float(args.early_stop_min_delta)):
            best_metric_val = curr_cmp
            improved = True
            ckpt_path = os.path.join(args.ckpt_dir, "backbone_best.pt")
            torch.save(backbone.state_dict(), ckpt_path)
            print(f"  ✓ saved best by {args.best_metric}: {ckpt_path} (value={curr if curr==curr else 'nan'})")
        # Early stopping bookkeeping
        if args.early_stop_patience and args.early_stop_patience > 0:
            if improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stop_patience:
                    print(f"  ↳ early stopping: no improvement in {args.early_stop_patience} epochs (best {args.best_metric}={best_metric_val:.6f})")
                    break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.expanduser("~/myprojects/faceid-project/data/train"))
    ap.add_argument("--val",   default=os.path.expanduser("~/myprojects/faceid-project/data/val"))
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/myprojects/faceid-project/models"))
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pairs_per_epoch", type=int, default=50000)
    ap.add_argument("--pos_ratio", type=float, default=0.5)
    ap.add_argument("--eval_pairs", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet-50 and ImageNet normalization")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0, help="Freeze ResNet backbone for the first N epochs")
    ap.add_argument("--progress_step_pct", type=float, default=1.0, help="Print progress every X percent; e.g., 1.0 -> 1%, 0.1 -> 0.1%, 0 to disable")
    ap.add_argument("--scale_logits", type=float, default=20.0, help="Scale factor for cosine logits in BCE loss (e.g., 20, 32, 64)")
    ap.add_argument("--backbone_lr_scale", type=float, default=0.1, help="Relative LR for backbone vs head after unfreeze (e.g., 0.1)")
    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none", help="LR scheduler to use")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs for LR scheduler")
    ap.add_argument("--train_mode", choices=["pair", "arcface"], default="pair", help="Training objective: pair BCE or ArcFace classification")
    ap.add_argument("--best_metric", choices=["acc","auc","tar_1e-2","tar_1e-3"], default="acc", help="Metric to select best checkpoint")
    ap.add_argument("--early_stop_patience", type=int, default=0, help="Stop early after N epochs with no improvement (0=disable)")
    ap.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum metric delta to qualify as improvement")
    ap.add_argument("--arcface_s", type=float, default=64.0, help="ArcFace scale s (logit multiplier)")
    ap.add_argument("--arcface_m", type=float, default=0.5, help="ArcFace margin m (radians)")
    ap.add_argument("--clip_grad_norm", type=float, default=0.0, help="Max global grad norm; 0 to disable")
    args = ap.parse_args()
    main(args)
