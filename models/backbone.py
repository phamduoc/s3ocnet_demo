import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch import nn

def get_backbone(name="resnet18"):
    if name == "resnet18":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # Loại bỏ tầng fully-connected để lấy đặc trưng (512-dim)
    model.fc = nn.Identity()
    return model