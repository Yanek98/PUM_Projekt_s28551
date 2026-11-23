import io
import logging
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torchvision import models, transforms

from src.automl.tracking import candidate
from src.common import labels


SIZE = (96, 160)
EPOCHS = 7
PREPROCESS = transforms.Compose([
    transforms.Resize(SIZE), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def backbone(name, pretrained):
    if name == "MobileNetV3Small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        net.classifier = nn.Identity()
    elif name == "ShuffleNetV2":
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        net = models.shufflenet_v2_x0_5(weights=weights)
        net.fc = nn.Identity()
    else:
        raise ValueError(name)
    return net.eval()


def classifier(n_features, n_labels, hidden):
    if hidden:
        return nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.25), nn.Linear(hidden, n_labels))
    return nn.Linear(n_features, n_labels)


@torch.inference_mode()
def extract(archive, manifest, networks):
    chunks = {name: [] for name in networks}
    for start in range(0, len(manifest), 64):
        images = []
        for path in manifest.path.iloc[start:start + 64]:
            with Image.open(io.BytesIO(archive.read(path))) as image:
                images.append(PREPROCESS(image.convert("RGB")))
        batch = torch.stack(images)
        for name, network in networks.items():
            chunks[name].append(network(batch))
        if start % 4096 == 0:
            logging.info("ImageNet embeddings %d/%d", start, len(manifest))
    return {name: torch.cat(parts) for name, parts in chunks.items()}


def fit_head(x, y, hidden):
    torch.manual_seed(42)
    head = classifier(x.shape[1], y.shape[1], hidden)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    objective = nn.BCEWithLogitsLoss()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for rows in torch.randperm(len(y)).split(256):
            optimizer.zero_grad()
            loss = objective(head(x[rows]), y[rows])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        mlflow.log_metric("train_loss", total / len(y), step=epoch)
    return head.eval()


def train(path, execution):
    torch.set_num_threads(3)
    networks = {name: backbone(name, True) for name in ("MobileNetV3Small", "ShuffleNetV2")}
    with TemporaryDirectory() as tmp:
        # A sequential copy avoids thousands of seeks across a Windows bind mount.
        local = Path(tmp) / "images.zip"
        shutil.copyfile(path, local)
        with zipfile.ZipFile(local) as archive:
            manifest = pd.read_csv(archive.open("manifest.csv"))
            embeddings = extract(archive, manifest, networks)
    train_mask = manifest.split.eq("train").to_numpy()
    val_mask = manifest.split.eq("val").to_numpy()
    if set(manifest.app_id[train_mask]) & set(manifest.app_id[val_mask]) or not manifest.variant[val_mask].eq("original").all():
        raise ValueError("Invalid validation split for images")
    train_labels = manifest.genres[train_mask].map(labels)
    encoder = MultiLabelBinarizer()
    targets = torch.tensor(encoder.fit_transform(train_labels), dtype=torch.float32)
    classes = encoder.classes_.tolist()
    val_games = manifest[val_mask].groupby("app_id").genres.first().map(labels)
    extra = sorted({c for values in val_games for c in values} - set(classes))
    truth = MultiLabelBinarizer(classes=classes + extra).fit_transform(val_games)
    ids = []
    for name, vectors in embeddings.items():
        mean, std = vectors[train_mask].mean(0), vectors[train_mask].std(0).clamp_min(1e-5)
        train_x, val_x = (vectors[train_mask] - mean) / std, (vectors[val_mask] - mean) / std
        for hidden in (0, 128):
            label = f"{name}-{'MLP' if hidden else 'Linear'}"
            params = dict(backbone=name, pretrained="ImageNet1K", frozen_backbone=True, hidden=hidden,
                          epochs=EPOCHS, seed=42, lr=0.001, threshold=0.5, image_size=str(SIZE),
                          train_images=len(train_x), val_images=len(val_x), val_games=len(truth),
                          train_labels=len(classes), val_only_labels=len(extra))
            with candidate("images", execution, label, params) as rid, TemporaryDirectory() as tmp:
                head = fit_head(train_x, targets, hidden)
                with torch.no_grad():
                    prob = torch.sigmoid(head(val_x)).numpy()
                avg = pd.DataFrame(prob, index=manifest.app_id[val_mask]).groupby(level=0).mean().loc[val_games.index]
                prediction = np.pad(avg.to_numpy() >= 0.5, ((0, 0), (0, len(extra))))
                mlflow.log_metrics({f"f1_{a}": f1_score(truth, prediction, average=a, zero_division=0)
                                    for a in ("micro", "macro")})
                file = Path(tmp) / "weights.pt"
                torch.save(dict(backbone=name, backbone_state=networks[name].state_dict(), head_state=head.state_dict(),
                                mean=mean, std=std, hidden=hidden, classes=classes, size=SIZE), file)
                mlflow.log_artifact(str(file), "model")
                ids.append(rid)
    return ids
