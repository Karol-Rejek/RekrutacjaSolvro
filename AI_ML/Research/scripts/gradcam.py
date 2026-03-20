# Wymaga: pip install grad-cam
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_scripted_for_gradcam(pt_path: str):
    """TorchScript → eager model (GradCAM wymaga eager mode)."""
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.conv1    = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc       = torch.nn.Linear(512, 2)
    # Wczytaj wagi ze state_dict zapisanego osobno
    model.load_state_dict(torch.load(pt_path.replace(".pt", "_state.pth")))
    model.eval()
    return model

def run_gradcam(model_path: str, img_path: str, label: str):
    model        = load_scripted_for_gradcam(model_path)
    target_layer = [model.layer4[-1]]  # ostatni blok ResNet

    img_pil = Image.open(img_path).convert("L").resize((64, 64))
    img_np  = np.array(img_pil) / 255.0
    tensor  = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with GradCAM(model=model, target_layers=target_layer) as cam:
        mask = cam(input_tensor=tensor)[0]

    img_rgb = np.stack([img_np] * 3, axis=-1).astype(np.float32)
    vis     = show_cam_on_image(img_rgb, mask, use_rgb=True)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1); plt.imshow(img_np, cmap="gray"); plt.title("Oryginał")
    plt.subplot(1, 2, 2); plt.imshow(vis);                  plt.title(f"GradCAM: {label}")
    plt.savefig(f"plots/gradcam_{Path(img_path).stem}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    # Przykład użycia
    run_gradcam("models/resnet18.pt", "data/raw/circle/hand_photo/img_001.jpg", "HandDrawn")
    run_gradcam("models/resnet18.pt", "data/raw/circle/stamp_photo/img_001.jpg", "Digital")
