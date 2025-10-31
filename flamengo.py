

import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torchvision
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def rgb01_to_lab01(rgb01: np.ndarray) -> np.ndarray:
    rgb = rgb01.copy()
    mask = rgb <= 0.04045
    rgb[mask] = rgb[mask] / 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    XYZ = np.tensordot(rgb, M.T, axes=1)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = XYZ[..., 0] / Xn; Y = XYZ[..., 1] / Yn; Z = XYZ[..., 2] / Zn
    d = 6/29
    f = lambda t: np.where(t > d**3, np.cbrt(t), (t/(3*d*d) + 4/29))
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    L01 = (L/100.0).clip(0, 1)
    a01 = ((a + 86)/200.0).clip(0, 1)
    b01 = ((b + 107)/215.0).clip(0, 1)
    return np.stack([L01, a01, b01], axis=-1).astype(np.float32)

def build_features_lab_spatial(img01: np.ndarray, lambda_s: float = 1.0) -> np.ndarray:
    H, W = img01.shape[:2]
    lab = rgb01_to_lab01(img01)
    yy, xx = np.meshgrid(np.arange(H)/H, np.arange(W)/W, indexing="ij")
    yy = (yy.astype(np.float32) * lambda_s)[..., None]
    xx = (xx.astype(np.float32) * lambda_s)[..., None]
    feats = np.dstack([lab, yy, xx]).reshape(-1, 5) 
    return feats

def kmeans_np(X: np.ndarray, K=2, iters=12, seed=42):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], K, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :])**2).sum(-1)
        a = d.argmin(axis=1)
        for k in range(K):
            pts = X[a == k]
            if len(pts): C[k] = pts.mean(axis=0)
    return a.reshape(-1), C


def get_pet(split="test", img_size=256):
    ds = torchvision.datasets.OxfordIIITPet(
        root="./data", download=True, target_types=("segmentation",)
    )
    n = len(ds); idx = np.arange(n); np.random.seed(123); np.random.shuffle(idx)
    n_val = n//5; n_test = n//5
    pick = idx[:n-n_val-n_test] if split=="train" else idx[n-n_val-n_test:n-n_test] if split=="val" else idx[n-n_test:]

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(pick)
        def __getitem__(self, i):
            img, mask = ds[pick[i]]

            m = np.array(mask, dtype=np.uint8)
            pet = (m != 3).astype(np.uint8)

            img = TF.resize(img, [img_size, img_size], interpolation=InterpolationMode.BILINEAR)

            mask_pil = Image.fromarray((pet * 255).astype(np.uint8))  
            mask_rs = TF.resize(mask_pil, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
            mask_bin = (np.array(mask_rs) > 127).astype(np.uint8)    
            return img, torch.from_numpy(mask_bin).unsqueeze(0)     

    return Wrap()

def build_maskrcnn(num_classes=2, img_size=256, device="cpu", ckpt=None):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    inF = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(inF, num_classes)
    inFm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(inFm, 256, num_classes)
    model.transform.min_size = (img_size,)
    model.transform.max_size = img_size
    model.to(device).eval()
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    return model

def pred_fn_maskrcnn(model, img_pil, device="cpu"):
    t = TF.pil_to_tensor(img_pil).float()/255.0
    out = model([t.to(device)])[0]
    if "masks" in out and out["masks"].shape[0] > 0:
        m = out["masks"].detach().cpu().numpy()
        return (m > 0.5).astype(np.uint8).max(axis=0)[0]
    return np.zeros((t.shape[1], t.shape[2]), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ex.: results/checkpoints/maskrcnn_best.pt")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lambda-s", type=float, default=1.0)
    ap.add_argument("--kmeans-iters", type=int, default=12)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    set_seed(42)
    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    outdir = Path(args.out); ensure_dir(outdir)

    ds = get_pet(args.split, args.img_size)
    idx = int(np.clip(args.index, 0, len(ds)-1))
    img_pil, gt = ds[idx]
    gt_np = gt.squeeze(0).numpy()

    modelA = build_maskrcnn(2, args.img_size, device, args.ckpt)
    t0 = time.time(); predA = pred_fn_maskrcnn(modelA, img_pil, device); tA = time.time()-t0

    img01 = np.array(img_pil).astype(np.float32)/255.0
    feats = build_features_lab_spatial(img01, lambda_s=args.lambda_s)
    t0 = time.time(); a, _ = kmeans_np(feats, K=2, iters=args.kmeans_iters, seed=42); tB = time.time()-t0
    predB = a.reshape(img01.shape[0], img01.shape[1]).astype(np.uint8)
    if (((predB==0)&(gt_np==1)).sum() > ((predB==1)&(gt_np==1)).sum()):
        predB = 1 - predB

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(img_pil); ax[0].set_title("Imagem"); ax[0].axis("off")
    ax[1].imshow(gt_np, cmap="viridis"); ax[1].set_title("GT (pet=1)"); ax[1].axis("off")
    ax[2].imshow(predA, cmap="viridis"); ax[2].set_title(f"Mask R-CNN\n{tA*1000:.0f} ms"); ax[2].axis("off")
    ax[3].imshow(predB, cmap="viridis"); ax[3].set_title(f"K-means\n{tB*1000:.0f} ms"); ax[3].axis("off")
    fig.suptitle("Comparativo em uma imagem", fontsize=14); plt.tight_layout()
    out_png = Path(outdir)/f"demo_{args.split}_idx{idx}.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close()

    def iou_dice(pred, gt):
        pred = pred.astype(bool); gt = gt.astype(bool)
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = inter / (union + 1e-7)
        dice = (2*inter) / (pred.sum() + gt.sum() + 1e-7)
        return float(iou), float(dice)

    iouA, diceA = iou_dice(predA, gt_np)
    iouB, diceB = iou_dice(predB, gt_np)

    summary = {
        "image_index": idx,
        "times_ms": {"MaskRCNN": int(tA*1000), "KMeans": int(tB*1000)},
        "metrics_single_image": {
            "MaskRCNN": {"IoU": iouA, "Dice": diceA},
            "KMeans":   {"IoU": iouB, "Dice": diceB},
        },
        "figure": str(out_png)
    }
    ensure_dir(outdir)
    with open(Path(outdir)/f"demo_{args.split}_idx{idx}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResumo:", json.dumps(summary, indent=2))
    print(f"Figura salva em: {out_png}")

if __name__ == "__main__":
    main()