import torch                
import torch.nn as nn
from torchvision import models
try:
    # torchvision >= 0.13
    from torchvision.models import ResNet50_Weights  # type: ignore
except Exception:  # pragma: no cover - compatibility for older torchvision
    ResNet50_Weights = None  # type: ignore

class L2Norm(nn.Module):
    """L2-normalize feature vectors along channel dimension (dim=1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=1)

class BackboneResNet50(nn.Module):
    """ResNet-50 backbone producing L2-normalized embeddings of size `embed_dim`.

    The final fully-connected layer of ResNet-50 is removed and replaced with a
    linear projection to `embed_dim`, followed by L2 normalization.
    """

    def __init__(self, embed_dim: int = 512, channels_last: bool = True, pretrained: bool = False):
        super().__init__()
        # Build torchvision ResNet-50 backbone with/without pretrained weights.
        # Handle both new (weights=...) and old (pretrained=...) API styles.
        if pretrained:
            if ResNet50_Weights is not None:
                m = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # type: ignore
            else:
                m = models.resnet50(pretrained=True)
        else:
            try:
                m = models.resnet50(weights=None)
            except TypeError:
                m = models.resnet50(pretrained=False)
        in_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_dim, embed_dim, bias=False)
        self.norm = L2Norm()
        self.channels_last = channels_last

    def forward(self, x):
        # Optionally use channels_last memory format for potential speedups on GPU
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.backbone(x)
        emb = self.head(feat)
        emb = self.norm(emb)
        return emb

class ArcMarginProduct(nn.Module):
    """ArcFace head: Additive Angular Margin Softmax.

    Applies an additive angular margin m to the target class angle before scaling by s.
    Reference: ArcFace (CVPR 2019)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        # Precompute constants for efficiency
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159265) - torch.tensor(m))
        self.mm = torch.sin(torch.tensor(3.14159265) - torch.tensor(m)) * m

    def forward(self, input, label):
        # input: (N, in_features) normalized embeddings
        # label: (N,), integer class indices
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot labels to select phi for the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
