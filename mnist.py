"""Train on MNIST train split, evaluate on MNIST test split.
Models: MLP, CNN, Transformer encoder. Optional GPU random-affine augmentation.
Usage: python mnist.py --models mlp cnn transformer --aug none affine --seeds 0 1 2
"""
import argparse, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = 0.1307, 0.3081


def load(root="data"):
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        x = ds.data.float().div(255).unsqueeze(1)  # N,1,28,28 in [0,1]
        out += [x.to(DEV), ds.targets.to(DEV)]
    return out  # xtr, ytr, xte, yte


def norm(x):
    return (x - MEAN) / STD


def affine_aug(x, rot=15, trans=0.1, scale=(0.9, 1.1), shear=10):
    """Random rotation/translation/scale/shear applied per-sample on GPU."""
    n = x.size(0)
    r = (torch.rand(n, device=x.device) * 2 - 1) * rot * math.pi / 180
    sh = (torch.rand(n, device=x.device) * 2 - 1) * shear * math.pi / 180
    s = torch.empty(n, device=x.device).uniform_(*scale)
    t = (torch.rand(n, 2, device=x.device) * 2 - 1) * trans * 2  # grid coords span [-1,1]
    cos, sin = torch.cos(r) / s, torch.sin(r) / s
    theta = torch.stack([
        torch.stack([cos, -sin + torch.tan(sh) * cos, t[:, 0]], 1),
        torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class MLP(nn.Module):  # shallow: one hidden layer
    def __init__(s, h=512):
        super().__init__()
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(784, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, 10))
    def forward(s, x): return s.net(x)


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        s.net = nn.Sequential(*blk(1, 32), *blk(32, 32), nn.MaxPool2d(2),
                              *blk(32, 64), *blk(64, 64), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 10))
    def forward(s, x): return s.net(x)


class Transformer(nn.Module):  # ViT-style: 4x4 patches -> 49 tokens + CLS, encoder only
    def __init__(s, d=64, heads=4, layers=4, patch=4):
        super().__init__()
        s.embed = nn.Conv2d(1, d, patch, stride=patch)
        n = (28 // patch) ** 2
        s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, layers)
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 10))
    def forward(s, x):
        t = s.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1) + s.pos
        return s.head(s.enc(t)[:, 0])


MODELS = {"mlp": MLP, "cnn": CNN, "transformer": Transformer}


@torch.no_grad()
def evaluate(m, x, y):
    m.eval()
    correct = sum((m(norm(x[i:i + 2000])).argmax(1) == y[i:i + 2000]).sum().item() for i in range(0, len(x), 2000))
    return 100 * correct / len(x)


def run(name, aug, seed, data, epochs, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    m = MODELS[name]().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb = xtr[idx]
            if aug == "affine": xb = affine_aug(xb)
            loss = F.cross_entropy(m(norm(xb)), ytr[idx], label_smoothing=0.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    # test set is touched only once, after the last epoch (no early stopping / model picking on test)
    return evaluate(m, xtr, ytr), evaluate(m, xte, yte), sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--aug", nargs="+", default=["none", "affine"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    data = load()
    print(f"train={len(data[0])} test={len(data[2])} device={DEV}")
    res = []
    for name in a.models:
        for aug in a.aug:
            for seed in a.seeds:
                t = time.time()
                tr, te, npar = run(name, aug, seed, data, a.epochs)
                res.append(dict(model=name, aug=aug, seed=seed, params=npar, train_acc=tr, test_acc=te, secs=round(time.time() - t)))
                print(res[-1], flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
