import torch
import torchvision.models as models
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 2  
IMG_SIZE    = (1, 1, 64, 64)  

def export(name: str, model: torch.nn.Module) -> None:
    if hasattr(model, "conv1"):
        orig = model.conv1
        model.conv1 = torch.nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False
        )
    elif hasattr(model, "features"):
        orig = model.features[0][0]
        model.features[0][0] = torch.nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False
        )

    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    elif hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)

    model.eval()
    dummy = torch.zeros(IMG_SIZE)

    traced = torch.jit.trace(model, dummy)
    out_path = MODELS_DIR / f"{name}.pt"
    traced.save(str(out_path))
    print(f"[OK] Zapisano: {out_path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    export("resnet18",     models.resnet18(weights=models.ResNet18_Weights.DEFAULT))
    export("mobilenet_v2", models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT))
    print("Eksport zakończony.")
