import os, time, json, random, copy, argparse, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch, torchvision
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image

def get_args():
    p = argparse.ArgumentParser("Mask R-CNN vs Spatial K-means (rápido e completo)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam","sgd"])
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--kmeans-workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--lambda-s", type=float, default=0.5, help="peso do termo espacial (y/H,x/W)")
    p.add_argument("--kmeans-iters", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--out", type=str, default=None, help="pasta de saída. default: /content/results se existir, senão ./results")
    p.add_argument("--img-size-kmeans", type=int, default=0, help="(opcional) redimensiona SÓ para K-means; 0 = usar IMG_SIZE")
    return p.parse_args()

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class AvgMeter:
    def __init__(self): self.v=0.0; self.n=0
    def add(self, x, k=1): self.v += float(x)*k; self.n += k
    @property
    def avg(self): return self.v/max(1,self.n)

class PetSegBinary(torch.utils.data.Dataset):
    """Oxford-IIIT Pet -> binário: pet (1, inclui borda) vs fundo (0)."""
    def __init__(self, root, split='train', img_size=256, seed=42):
        self.ds = OxfordIIITPet(root=root, download=True, target_types=('segmentation',))
        self.img_size = img_size
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.to_tensor = transforms.ToTensor()
        n = len(self.ds); idxs = np.arange(n)
        rng = np.random.default_rng(seed); rng.shuffle(idxs)
        n_train = int(0.6*n); n_val = int(0.2*n)
        split_map = {'train': idxs[:n_train], 'val': idxs[n_train:n_train+n_val], 'test': idxs[n_train+n_val:]}
        self.indices = split_map[split]; self.split = split

    def _augment(self, img, mask):
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        if self.split=='train' and random.random()<0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img, seg = self.ds[idx]
        seg_np = np.array(seg, dtype=np.uint8)
        bin_mask = (seg_np != 3).astype(np.uint8) 
        img, seg_img = self._augment(img, Image.fromarray(bin_mask*255))
        img_t = self.norm(self.to_tensor(img))
        mask_t = torch.from_numpy(np.array(seg_img)//255).long()
        return img_t, mask_t

def collate_fn(batch):
    imgs, masks = zip(*batch)
    return list(imgs), list(masks)

NUM_CLASSES = 2
CLASSES = ["bg","pet"]

def fast_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    y_true = y_true.flatten(); y_pred = y_pred.flatten()
    mask = (y_true >= 0) & (y_true < num_classes)
    hist = np.bincount(num_classes*y_true[mask] + y_pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)
    return hist

def iou_dice_from_confmat(confmat):
    tp = np.diag(confmat).astype(np.float32)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    dice = (2*tp) / (2*tp + fp + fn + 1e-7)
    return iou, float(np.nanmean(iou)), dice, float(np.nanmean(dice))

def build_mask_rcnn(img_size, num_classes=2):
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    except TypeError:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.transform.min_size = (img_size,); model.transform.max_size = img_size

    rpn = model.rpn
    if hasattr(rpn, "pre_nms_top_n_train"): rpn.pre_nms_top_n_train = 100; rpn.pre_nms_top_n_test = 50
    if hasattr(rpn, "post_nms_top_n_train"): rpn.post_nms_top_n_train = 50;  rpn.post_nms_top_n_test = 25
    if hasattr(rpn, "batch_size_per_image"): rpn.batch_size_per_image = 128
    if hasattr(model.roi_heads, "batch_size_per_image"): model.roi_heads.batch_size_per_image = 128

    for p in model.backbone.parameters(): p.requires_grad = False
    return model

def masks_to_instances(masks, device):
    out = []
    mnp = masks.cpu().numpy()
    for m in mnp:
        ys, xs = np.where(m==1)
        if len(xs)==0:
            out.append({"boxes": torch.zeros((0,4), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        "masks": torch.zeros((0, m.shape[0], m.shape[1]), dtype=torch.uint8, device=device)})
            continue
        y1, y2 = ys.min(), ys.max(); x1, x2 = xs.min(), xs.max()
        box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=device)
        inst_mask = torch.from_numpy(m[None, ...].astype(np.uint8)).to(device)
        labels = torch.tensor([1], dtype=torch.int64, device=device)
        out.append({"boxes": box, "labels": labels, "masks": inst_mask})
    return out

def rgb01_to_lab01(rgb):
    a = 0.055
    rgb_lin = np.where(rgb <= 0.04045, rgb/12.92, ((rgb + a)/(1 + a))**2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    XYZ = np.tensordot(rgb_lin, M.T, axes=1)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = XYZ[...,0]/Xn, XYZ[...,1]/Yn, XYZ[...,2]/Zn
    eps = 216/24389; k = 24389/27
    def f(t): return np.where(t>eps, np.cbrt(t), (k*t + 16)/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16; a = 500*(fx - fy); b = 200*(fy - fz)
    L01 = np.clip(L/100.0, 0, 1); a01 = np.clip((a+110)/220, 0, 1); b01 = np.clip((b+110)/220, 0, 1)
    return np.stack([L01, a01, b01], axis=-1).astype(np.float32)

def build_features_lab_spatial(img_t, lambda_s=0.5, out_size=None):
    mean = torch.tensor([0.485,0.456,0.406], device=img_t.device)[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225], device=img_t.device)[:,None,None]
    rgb = (img_t*std + mean).clamp(0,1).permute(1,2,0).cpu().numpy()
    if out_size and (rgb.shape[0]!=out_size or rgb.shape[1]!=out_size):
        rgb = np.array(Image.fromarray((rgb*255).astype(np.uint8)).resize((out_size,out_size), Image.BILINEAR))/255.0
    H,W,_ = rgb.shape
    lab = rgb01_to_lab01(rgb)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pos = np.stack([yy/H, xx/W], axis=-1).astype(np.float32)
    feats = np.concatenate([lab, lambda_s*pos], axis=-1).reshape(-1,5).astype(np.float32)
    return feats, (H,W)

def kmeans_np(X, K=2, iters=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], K, replace=False)
    centroids = X[idx].copy()
    for _ in range(iters):
        dists = ((X[:,None,:]-centroids[None,:,:])**2).sum(-1)
        assign = dists.argmin(axis=1)
        for k in range(K):
            pts = X[assign==k]
            if len(pts)>0: centroids[k] = pts.mean(axis=0)
    return assign, centroids

def kmeans_spatial_predict_np(img_t, iters, seed, lambda_s, out_size=None):
    feats, (H,W) = build_features_lab_spatial(img_t, lambda_s=lambda_s, out_size=out_size)
    assign,_ = kmeans_np(feats, K=2, iters=iters, seed=seed)
    seg = assign.reshape(H,W).astype(np.uint8)
    return seg

def evaluate_loader(pred_fn, loader, device, model_name=""):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for imgs, gts in loader:
            img = imgs[0].to(device, non_blocking=True)
            pr = pred_fn([img])[0].cpu().numpy().astype(np.int64)
            gt = gts[0].cpu().numpy().astype(np.int64)
            conf += fast_confusion_matrix(gt, pr, NUM_CLASSES)
    iou, miou, dice, mdice = iou_dice_from_confmat(conf)
    print(f"[{model_name}] Test mIoU={miou:.4f} | mDice={mdice:.4f}")
    return {"confusion_matrix": conf.tolist(),
            "iou_per_class": iou.tolist(),
            "miou": miou,
            "dice_per_class": dice.tolist(),
            "mdice": mdice}

@torch.no_grad()
def evaluate_mask_for_val(model, loader, device):
    model.eval(); conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for imgs, gts in loader:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        outputs = model(imgs)
        preds = []
        for out in outputs:
            if 'masks' in out and out['masks'].shape[0] > 0:
                pm = (out['masks'].squeeze(1) > 0.5).float()
                comb = (pm.max(dim=0)[0] > 0.5).long()
            else:
                comb = torch.zeros(imgs[0].shape[1:], dtype=torch.long, device=device)
            preds.append(comb)
        gt_np = torch.stack(gts).cpu().numpy().astype(np.int64)
        pr_np = torch.stack(preds).cpu().numpy().astype(np.int64)
        conf += fast_confusion_matrix(gt_np, pr_np, NUM_CLASSES)
    iou, miou, dice, mdice = iou_dice_from_confmat(conf)
    return float(miou), float(mdice), conf

def eval_per_image(pred_fn, loader, device):
    conf_total = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int64)
    rows = []
    idx = 0
    with torch.no_grad():
        for imgs, gts in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            pred = pred_fn(imgs) 
            pred = pred.detach().cpu().numpy().astype(np.int64)
            gt   = torch.stack(gts).cpu().numpy().astype(np.int64)
            for k in range(pred.shape[0]):
                c = fast_confusion_matrix(gt[k], pred[k], NUM_CLASSES)
                iou_c, miou, dice_c, mdice = iou_dice_from_confmat(c)
                rows.append({
                    "img_idx": idx,
                    "IoU_bg": float(iou_c[0]), "IoU_pet": float(iou_c[1]), "mIoU": float(miou),
                    "Dice_bg": float(dice_c[0]), "Dice_pet": float(dice_c[1]), "mDice": float(mdice)
                })
                conf_total += c
                idx += 1
    df = pd.DataFrame(rows)
    return df, conf_total

def save_boxplot(series_list, labels, title, out_png):
    plt.figure()
    plt.boxplot(series_list, labels=labels, showfliers=True)
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png); plt.show()

def save_bar(vals, labels, title, out_png, ylabel=""):
    plt.figure()
    xs = range(len(vals))
    plt.bar(xs, vals); plt.xticks(xs, labels)
    if ylabel: plt.ylabel(ylabel)
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png); plt.show()

def save_confusion_heatmap(conf, title, out_png):
    conf = conf.astype(np.float64)
    conf_norm = conf / (conf.sum(axis=1, keepdims=True)+1e-7)
    plt.figure()
    plt.imshow(conf_norm, aspect='auto'); plt.colorbar()
    plt.xticks(range(NUM_CLASSES), CLASSES); plt.yticks(range(NUM_CLASSES), CLASSES)
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png); plt.show()

def show_examples_grid(predA_fn, predB_fn, dataset, device, k=5, seed=42, out_png=None):
    random.seed(seed)
    idxs = random.sample(range(len(dataset)), k)
    ncols = 4; nrows = k
    figsize=(9, 2.2*k)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    mean = np.array([0.485, 0.456, 0.406]); std  = np.array([0.229, 0.224, 0.225])
    for r, idx in enumerate(idxs):
        img_t, mask_t = dataset[idx]
        img = (img_t.permute(1,2,0).cpu().numpy()*std + mean).clip(0,1)
        with torch.no_grad():
            pA = predA_fn([img_t.to(device)])[0].cpu().numpy()
            pB = predB_fn([img_t.to(device)])[0].cpu().numpy()
        axes[r,0].imshow(img);      axes[r,0].set_title("Imagem"); axes[r,0].axis('off')
        axes[r,1].imshow(mask_t, vmin=0, vmax=1); axes[r,1].set_title("GT"); axes[r,1].axis('off')
        axes[r,2].imshow(pA, vmin=0, vmax=1);     axes[r,2].set_title("Pred A"); axes[r,2].axis('off')
        axes[r,3].imshow(pB, vmin=0, vmax=1);     axes[r,3].set_title("Pred B"); axes[r,3].axis('off')
    plt.suptitle("Exemplos qualitativos"); plt.tight_layout(rect=[0,0,1,0.97])
    if out_png: plt.savefig(out_png)
    plt.show()

def main():
    args = get_args()
    set_seed(args.seed)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(min(4, os.cpu_count() or 1))

    if args.out is None:
        out_dir = Path("/content/results") if Path("/content").exists() else Path("results")
    else:
        out_dir = Path(args.out)
    (out_dir/"logs").mkdir(parents=True, exist_ok=True)
    (out_dir/"checkpoints").mkdir(exist_ok=True)
    (out_dir/"figs").mkdir(exist_ok=True)
    (out_dir/"metrics").mkdir(exist_ok=True)

    print(f"Torch {torch.__version__} | Torchvision {torchvision.__version__} | device={device}")
    print(f"DL workers={args.num_workers} | KMeans threads={args.kmeans_workers}")

    root = Path("data")
    train_set = PetSegBinary(root=str(root), split='train', img_size=args.img_size, seed=args.seed)
    val_set   = PetSegBinary(root=str(root), split='val',   img_size=args.img_size, seed=args.seed)
    test_set  = PetSegBinary(root=str(root), split='test',  img_size=args.img_size, seed=args.seed)

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=(args.num_workers>0), prefetch_factor=2)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **dl_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, **dl_kwargs)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1, shuffle=False, **{k:v for k,v in dl_kwargs.items() if k!='collate_fn'})

    modelA = build_mask_rcnn(args.img_size).to(device)
    params = [p for p in modelA.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizerA = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    else:
        optimizerA = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    schedulerA = optim.lr_scheduler.ReduceLROnPlateau(optimizerA, mode='max', factor=0.5, patience=1)
    scaler = GradScaler(enabled=use_cuda)

    def pred_fn_maskrcnn(img_batch):
        modelA.eval()
        imgs = [im.to(device, non_blocking=True) for im in img_batch]
        outputs = modelA(imgs)
        preds = []
        for out in outputs:
            if 'masks' in out and out['masks'].shape[0] > 0:
                pm = (out['masks'].squeeze(1) > 0.5).float()
                comb = (pm.max(dim=0)[0] > 0.5).long()
            else:
                H, W = imgs[0].shape[1], imgs[0].shape[2]
                comb = torch.zeros((H,W), dtype=torch.long, device=imgs[0].device)
            preds.append(comb)
        return torch.stack(preds)

    history = {"epoch":[], "train_loss":[], "val_mIoU":[], "val_mDice":[], "epoch_time_s":[]}
    no_improve = 0; best_miou = -1; best_state = None
    t0_total = time.time()
    for epoch in range(1, args.epochs+1):
        modelA.train(); ep_t0 = time.time(); tr_loss = 0.0; steps = 0
        for imgs, masks in train_loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = masks_to_instances(torch.stack(masks).to(device, non_blocking=True), device)
            optimizerA.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                loss_dict = modelA(imgs, targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizerA); scaler.update()
            tr_loss += loss.item(); steps += 1
        ep_time = time.time()-ep_t0
        miou_val, mdice_val, _ = evaluate_mask_for_val(modelA, val_loader, device)
        print(f"[A][Epoch {epoch}] train_loss={tr_loss/max(1,steps):.4f} | val_mIoU={miou_val:.4f} | time={ep_time:.1f}s")
        history["epoch"].append(epoch); history["train_loss"].append(tr_loss/max(1,steps))
        history["val_mIoU"].append(float(miou_val)); history["val_mDice"].append(float(mdice_val))
        history["epoch_time_s"].append(float(ep_time))
        schedulerA.step(miou_val)
        if miou_val>best_miou:
            best_miou=miou_val; best_state=copy.deepcopy(modelA.state_dict()); no_improve=0
            torch.save(best_state, str(out_dir/'checkpoints'/'maskrcnn_best.pt'))
        else:
            no_improve+=1
        if no_improve>=args.patience:
            print("[A] Early stopping acionado."); break
    total_timeA=time.time()-t0_total
    if best_state is not None: modelA.load_state_dict(best_state)

    resA = evaluate_loader(lambda ims: pred_fn_maskrcnn(ims), test_loader, device, model_name="Mask R-CNN")

    def kmeans_single(img_cpu):
        return kmeans_spatial_predict_np(img_cpu, iters=args.kmeans_iters, seed=args.seed, lambda_s=args.lambda_s,
                                         out_size=(args.img_size_kmeans if args.img_size_kmeans>0 else None))

    confB = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    t0 = time.time()
    imgs_list, gts_list = [], []
    for imgs, gts in test_loader:
        imgs_list.append(imgs[0].cpu())
        gts_list.append(gts[0].cpu().numpy())

    with ThreadPoolExecutor(max_workers=args.kmeans_workers) as ex:
        results = list(ex.map(kmeans_single, imgs_list))

    for gt_np, pred in zip(gts_list, results):
        inter = np.logical_and(gt_np==1, pred==1).sum()
        union = np.logical_or(gt_np==1, pred==1).sum()+1e-7
        inter_fl = np.logical_and(gt_np==1, (1-pred)==1).sum()
        union_fl = np.logical_or(gt_np==1, (1-pred)==1).sum()+1e-7
        if inter/union < inter_fl/union_fl:
            pred = 1 - pred
        confB += fast_confusion_matrix(gt_np, pred, NUM_CLASSES)

    timeB = time.time()-t0
    iouB, miouB, diceB, mdiceB = iou_dice_from_confmat(confB)
    resB = {"confusion_matrix": confB.tolist(),
            "iou_per_class": iouB.tolist(),
            "miou": float(miouB),
            "dice_per_class": diceB.tolist(),
            "mdice": float(mdiceB)}
    print(f"[K-means] Test mIoU={miouB:.4f} | mDice={mdiceB:.4f} | tempo={timeB:.1f}s")

    summary = {
        "hyperparams": {"epochs":args.epochs,"batch_size":args.batch_size,"img_size":args.img_size,
                        "lr":args.lr,"optimizer":args.optimizer,"patience":args.patience,
                        "lambda_s":args.lambda_s,"kmeans_iters":args.kmeans_iters,
                        "kmeans_workers":args.kmeans_workers,"num_workers":args.num_workers,
                        "img_size_kmeans": args.img_size_kmeans},
        "device": str(device), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "results": {"ModelA_MaskRCNN": resA, "ModelB_SpatialKMeans": resB},
        "times": {"ModelA_train_total_sec": float(total_timeA), "ModelB_infer_total_sec": float(timeB)},
        "best_checkpoints": {"ModelA": str(out_dir/'checkpoints'/'maskrcnn_best.pt')}
    }
    with open(out_dir/'summary.json','w') as f: json.dump(summary, f, indent=2)

    with open(out_dir/'logs'/'maskrcnn_log.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","val_mIoU","val_mDice","epoch_time_s"])
        for e,tl,vm,vd,ts in zip(history["epoch"],history["train_loss"],history["val_mIoU"],history["val_mDice"],history["epoch_time_s"]):
            w.writerow([e,tl,vm,vd,ts])

    try:
        log_df = pd.read_csv(out_dir/'logs'/'maskrcnn_log.csv')
        if {"epoch","train_loss","val_mIoU"}.issubset(set(log_df.columns)):
            plt.figure(); plt.plot(log_df["epoch"], log_df["train_loss"], marker="o")
            plt.title("Treino — Loss por época (Modelo A)"); plt.xlabel("época"); plt.ylabel("train_loss"); plt.tight_layout()
            plt.savefig(out_dir/'figs'/'plot_loss_A.png'); plt.show()

            plt.figure(); plt.plot(log_df["epoch"], log_df["val_mIoU"], marker="o")
            plt.title("Val — mIoU por época (Modelo A)"); plt.xlabel("época"); plt.ylabel("val_mIoU"); plt.tight_layout()
            plt.savefig(out_dir/'figs'/'plot_val_miou_A.png'); plt.show()

            if "val_mDice" in log_df.columns:
                plt.figure(); plt.plot(log_df["epoch"], log_df["val_mDice"], marker="o")
                plt.title("Val — mDice por época (Modelo A)"); plt.xlabel("época"); plt.ylabel("val_mDice"); plt.tight_layout()
                plt.savefig(out_dir/'figs'/'plot_val_mdice_A.png'); plt.show()
    except Exception as e:
        print("[warn] não consegui plotar curvas:", e)

    print("[*] Gerando comparação estendida...")
    test_loader_A = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader_B = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    def pred_fn_kmeans(img_batch):
        preds = []
        for im in img_batch:
            seg = kmeans_spatial_predict_np(im.cpu(), iters=args.kmeans_iters, seed=args.seed,
                                            lambda_s=args.lambda_s, out_size=(args.img_size_kmeans if args.img_size_kmeans>0 else None))
            preds.append(torch.from_numpy(seg).to(device))
        return torch.stack(preds)

    dfA, confA = eval_per_image(lambda ims: pred_fn_maskrcnn(ims), test_loader_A, device)
    dfB, confB = eval_per_image(lambda ims: pred_fn_kmeans(ims),  test_loader_B, device)
    dfA.to_csv(out_dir/'metrics'/'ious_por_imagem_modelA.csv', index=False)
    dfB.to_csv(out_dir/'metrics'/'ious_por_imagem_modelB.csv', index=False)

    iouA_c, miouA, diceA_c, mdiceA = iou_dice_from_confmat(confA)
    iouB_c, miouB2, diceB_c, mdiceB2 = iou_dice_from_confmat(confB)
    delta_miou  = miouA - miouB2
    delta_mdice = mdiceA - mdiceB2
    winner = "Modelo A (Mask R-CNN)" if delta_miou >= 0 else "Modelo B (K-means)"
    print(f"Vencedor por mIoU (reavaliação por imagem): {winner} | Δ mIoU={delta_miou:.4f} | Δ mDice={delta_mdice:.4f}")

    try:
        with open(out_dir/'summary.json','r') as f: summ = json.load(f)
    except:
        summ = {"results":{}}
    summ["results"]["winner_by_mIoU"] = winner
    summ["results"]["delta_mIoU"] = float(delta_miou)
    summ["results"]["delta_mDice"] = float(delta_mdice)
    with open(out_dir/'summary.json','w') as f: json.dump(summ, f, indent=2)

    

    save_bar([miouA, miouB2], ["A (Mask R-CNN)","B (K-means)"], "mIoU (reavaliação por imagem)", out_dir/'figs'/'miou_global_reval.png', "mIoU")
    save_bar([mdiceA, mdiceB2], ["A (Mask R-CNN)","B (K-means)"], "mDice (reavaliação por imagem)", out_dir/'figs'/'mdice_global_reval.png', "mDice")
    save_bar(iouA_c, CLASSES, "IoU por classe — Modelo A", out_dir/'figs'/'iou_classes_A.png', "IoU")
    save_bar(iouB_c, CLASSES, "IoU por classe — Modelo B", out_dir/'figs'/'iou_classes_B.png', "IoU")
    save_bar((iouA_c - iouB_c).tolist(), CLASSES, "Δ IoU (A - B) por classe", out_dir/'figs'/'delta_iou_por_classe.png', "Δ IoU")

    save_boxplot([dfA["IoU_bg"], dfA["IoU_pet"]], ["bg(A)","pet(A)"], "Distribuição IoU por classe — Modelo A", out_dir/'figs'/'boxplot_iou_modelA.png')
    save_boxplot([dfB["IoU_bg"], dfB["IoU_pet"]], ["bg(B)","pet(B)"], "Distribuição IoU por classe — Modelo B", out_dir/'figs'/'boxplot_iou_modelB.png')

    save_confusion_heatmap(confA, "Matriz de confusão (normalizada) — Modelo A", out_dir/'figs'/'confusion_A.png')
    save_confusion_heatmap(confB, "Matriz de confusão (normalizada) — Modelo B", out_dir/'figs'/'confusion_B.png')

    show_examples_grid(lambda ims: pred_fn_maskrcnn(ims), lambda ims: pred_fn_kmeans(ims), test_set, device,
                       k=5, seed=args.seed, out_png=out_dir/'figs'/'qualitativos_inline.png')

    print("\\n===== RESULTADOS FINAIS (TESTE) =====")
    print(f"Modelo A (Mask R-CNN):     mIoU={resA['miou']:.4f} | mDice={resA['mdice']:.4f}")
    print(f"Modelo B (Spatial K-means): mIoU={resB['miou']:.4f} | mDice={resB['mdice']:.4f}")
    print("Melhor (mIoU):", "Modelo A" if resA['miou']>=resB['miou'] else "Modelo B")
    print("Arquivos em:", out_dir.resolve())

if __name__ == "__main__":
    main()
