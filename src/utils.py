import os, random, math, time
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cosine_sim(a: torch.Tensor, b: torch.Tensor):
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a @ b.t())

def compute_verif_metrics(embeds, labels, max_pairs=5000, far_targets=(1e-2, 1e-3)):
    """
    Verification metrics with balanced pair sampling and additional indicators.

    Args:
        embeds: torch.Tensor (N, D), L2-normalized embeddings or arbitrary.
        labels: torch.Tensor (N,), integer class ids corresponding to embeds.
        max_pairs: total number of pairs to evaluate (balanced pos/neg if possible).
        far_targets: tuple of FAR targets to report TAR@FAR for.

    Returns:
        dict with keys:
          - pairs, pos_pairs, neg_pairs
          - best_acc, best_th (threshold maximizing accuracy on sampled pairs)
          - auc (ROC-AUC over sampled pairs)
          - tar_at_far: { str(FAR): {"tar": float, "th": float} }
    """
    device = embeds.device
    N = embeds.size(0)
    # Subsample to keep eval affordable
    idx = torch.randperm(N, device=device)[:min(N, 8192)]
    E = embeds[idx]
    L = labels[idx]

    # Build class -> indices mapping
    class_to_indices = {}
    for i in range(L.numel()):
        cls = int(L[i].item())
        class_to_indices.setdefault(cls, []).append(i)
    classes = list(class_to_indices.keys())

    # Target balanced pairs
    total_pairs = int(max_pairs)
    n_pos_target = total_pairs // 2
    n_neg_target = total_pairs - n_pos_target

    pos_pairs = []
    neg_pairs = []

    # Sample positive pairs (two different indices from same class)
    if len(classes) > 0:
        # keep only classes with >= 2 items
        elig = [c for c in classes if len(class_to_indices[c]) >= 2]
        while len(pos_pairs) < n_pos_target and len(elig) > 0:
            c = random.choice(elig)
            i, j = random.sample(class_to_indices[c], 2)
            pos_pairs.append((i, j, 1))

    # Sample negative pairs (two different classes)
    if len(classes) >= 2:
        while len(neg_pairs) < n_neg_target:
            c1, c2 = random.sample(classes, 2)
            i = random.choice(class_to_indices[c1])
            j = random.choice(class_to_indices[c2])
            neg_pairs.append((i, j, 0))

    # Fallback: if not enough pos/neg due to data scarcity, sample random pairs
    all_pairs = pos_pairs + neg_pairs
    tries = 0
    max_tries = 5 * total_pairs
    while len(all_pairs) < total_pairs and tries < max_tries:
        i, j = random.sample(range(E.size(0)), 2)
        same = 1 if L[i] == L[j] else 0
        all_pairs.append((i, j, int(same)))
        tries += 1

    # Compute cosine similarities for sampled pairs
    if len(all_pairs) == 0:
        return {"pairs": 0, "pos_pairs": 0, "neg_pairs": 0, "best_acc": 0.0, "best_th": 0.0, "auc": float("nan"), "tar_at_far": {}}

    i1 = torch.tensor([p[0] for p in all_pairs], device=device, dtype=torch.long)
    i2 = torch.tensor([p[1] for p in all_pairs], device=device, dtype=torch.long)
    with torch.no_grad():
        sims_t = torch.nn.functional.cosine_similarity(E[i1], E[i2], dim=1)
    sims = sims_t.detach().cpu().numpy()
    gts = np.array([p[2] for p in all_pairs], dtype=np.int32)

    # Best accuracy by threshold grid
    best_acc, best_th = 0.0, 0.0
    for th in np.linspace(-1.0, 1.0, 401):
        pred = (sims >= th).astype(np.int32)
        acc = (pred == gts).mean()
        if acc > best_acc:
            best_acc, best_th = acc, float(th)

    # ROC computation (TPR vs FPR) and AUC
    pos_mask = gts == 1
    neg_mask = gts == 0
    P = int(pos_mask.sum())
    Nn = int(neg_mask.sum())
    auc = float("nan")
    tar_at_far = {}
    if P > 0 and Nn > 0:
        order = np.argsort(sims)[::-1]
        sorted_scores = sims[order]
        sorted_labels = gts[order]
        tp = np.cumsum(sorted_labels == 1)
        fp = np.cumsum(sorted_labels == 0)
        TPR = tp / max(1, P)
        FPR = fp / max(1, Nn)
        # AUC via trapezoidal rule
        auc = float(np.trapz(TPR, FPR))

        # TAR@FAR targets
        for t in far_targets:
            # Standardize key format like '1e-2' instead of '1e-02'
            try:
                exp = int(round(np.log10(float(t))))
                key = f"1e{exp}"
            except Exception:
                key = f"{t}"

            idxs = np.where(FPR <= t)[0]
            if idxs.size > 0:
                k = idxs[-1]
                tar_at_far[key] = {"tar": float(TPR[k]), "th": float(sorted_scores[k])}
            else:
                tar_at_far[key] = {"tar": float("nan"), "th": float("nan")}

    return {
        "pairs": int(len(all_pairs)),
        "pos_pairs": int(sum(1 for _,_,s in all_pairs if s == 1)),
        "neg_pairs": int(sum(1 for _,_,s in all_pairs if s == 0)),
        "best_acc": float(best_acc),
        "best_th": float(best_th),
        "auc": float(auc),
        "tar_at_far": tar_at_far,
    }
