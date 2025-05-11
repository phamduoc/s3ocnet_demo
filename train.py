import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from dataset_loader import get_cifar10_loaders
from models.backbone import get_backbone
from models.soch_head import SOCHHead


def train():
    device = config["device"]
    train_loader, _ = get_cifar10_loaders(config["batch_size"])

    backbone = get_backbone(config["backbone"]).to(device)
    soch_head = SOCHHead(input_dim=512, num_clusters=config["num_clusters"]).to(device)

    optimizer = optim.Adam(list(backbone.parameters()) + list(soch_head.parameters()), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        backbone.train()
        soch_head.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)
            features = backbone(images)
            similarity = soch_head(features)

            # Tổ chức phân cụm tự giám sát: entropy loss (ví dụ)
            p = torch.softmax(similarity, dim=1)
            entropy_loss = -torch.mean(torch.sum(p * torch.log(p + 1e-10), dim=1))

            loss = entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {total_loss/len(train_loader):.4f}")

    # Lưu mô hình
    torch.save({
        'backbone': backbone.state_dict(),
        'soch_head': soch_head.state_dict(),
    }, config["model_path"])