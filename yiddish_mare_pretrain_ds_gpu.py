"""
Dataset that preloads all images into GPU memory.
Use when the dataset fits in GPU RAM to avoid disk I/O during training.
Expects images already at 32x512 (no resize or padding).
"""
import os
import cv2
import torch
from torch.utils.data import Dataset


def _load_image(path):
    """Load grayscale image, normalize to [0, 1], return (1, H, W) float tensor."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    out = torch.from_numpy(img).float() / 255.0
    return out.unsqueeze(0)  # (1, H, W)


class YiddishMAEPretrainDatasetGPU(Dataset):
    """
    Loads all images at init time and keeps them on the given device (e.g. 'cuda').
    __getitem__ only indexes into the pre-allocated tensor, so no disk I/O during training.
    Images are assumed already 32x512; no preprocessing except normalize to [0, 1].
    """

    def __init__(self, image_folder, device="cuda"):
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"Image folder not found: {os.path.abspath(image_folder)}. "
                "Create the folder and add .png/.jpg/.jpeg/.tiff images, or fix the path."
            )
        paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))
        ]
        if len(paths) == 0:
            raise ValueError(
                f"No images found in {os.path.abspath(image_folder)}. "
                "Add .png, .jpg, .jpeg or .tiff files to the folder."
            )
        self.device = device

        tensors = []
        valid_paths = []
        for path in paths:
            t = _load_image(path)
            if t is not None:
                tensors.append(t)
                valid_paths.append(path)
        if len(tensors) == 0:
            raise ValueError(
                f"Could not load any image from {image_folder}. "
                "Check file formats and permissions."
            )
        # Stack: list of (1, H, W) -> (N, 1, H, W)
        self._data = torch.stack(tensors, dim=0).to(device)
        self.image_paths = valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self._data[idx]
