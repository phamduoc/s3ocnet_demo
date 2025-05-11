import torch

config = {
    "batch_size": 128,
    "num_epochs": 200,
    "learning_rate": 1e-3,
    "num_clusters": 10,  # CIFAR-10
    "backbone": "resnet18",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": "./checkpoints/s3ocnet_cifar10.pth"
}