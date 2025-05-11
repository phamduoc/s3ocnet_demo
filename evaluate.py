import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from config import config
from dataset_loader import get_cifar10_loaders
from models.backbone import get_backbone


def extract_features(backbone, dataloader, device):
    backbone.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            feats = backbone(imgs).cpu().numpy()
            features.append(feats)
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)


def evaluate():
    device = config["device"]
    train_loader, test_loader = get_cifar10_loaders(config["batch_size"])

    # Load backbone
    backbone = get_backbone(config["backbone"]).to(device)
    checkpoint = torch.load(config["model_path"], map_location=device)
    backbone.load_state_dict(checkpoint['backbone'])

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # Extract features
    X_train, y_train = extract_features(backbone, train_loader, device)
    X_test, y_test = extract_features(backbone, test_loader, device)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    # Logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, clf.predict(X_test))

    print(f"KNN Accuracy: {acc_knn*100:.2f}%")
    print(f"Linear Classifier Accuracy: {acc_lr*100:.2f}%")