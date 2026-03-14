import os
import time
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torch.profiler
from accelerate import Accelerator
from models_mae import MaskedAutoencoderViT
from yiddish_mare_pretrain_ds import YiddishSharedInRamDataset

IMG_SIZE = (32, 512)
LOG_DIR = "runs/mae_yiddish"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
# Prefer this file for TensorBoard reconstruction when present in lines_dir
PREFERRED_MONITOR_IMAGE = "BN_523.715_0013.tsv.processed_LINE_5.TIF"


def find_monitor_image(lines_dir):
    """Return path to monitor image: preferred file if present, else first image in lines_dir."""
    if not os.path.isdir(lines_dir):
        return None
    preferred_path = os.path.join(lines_dir, PREFERRED_MONITOR_IMAGE)
    if os.path.isfile(preferred_path):
        return preferred_path
    names = [
        f for f in os.listdir(lines_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not names:
        return None
    names.sort()
    return os.path.join(lines_dir, names[0])


def load_monitor_image(path, img_size, device):
    """Load one image for TensorBoard reconstruction logging; resize to img_size (H, W)."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().div(255.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return x.to(device)


def log_reconstruction(writer, model, monitor_img, epoch, mask_ratio=0.75):
    """Run model on monitor image and log original + reconstructed image to TensorBoard."""
    m = model.module if hasattr(model, "module") else model
    model.eval()
    with torch.no_grad():
        _, pred, _ = model(monitor_img, mask_ratio=mask_ratio)
        recon = m.unpatchify(pred)  # (1, 1, H, W)
    model.train()
    # Preds may be in normalized space when norm_pix_loss=True; scale to [0,1] for display
    r = recon[0].cpu().float()
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    orig = monitor_img[0].cpu().float()
    writer.add_image("monitor/original", orig, epoch, dataformats="CHW")
    writer.add_image("monitor/reconstructed", r, epoch, dataformats="CHW")


def train():
    # 1. Inicjalizacja Accelerate z wymuszeniem mixed_precision="bf16"
    accelerator = Accelerator(mixed_precision="bf16")
    
    # 2. Konfiguracja modelu pod parametry 32x512 i 1 kanał
    # Wykorzystujemy architekturę ViT-Base z Twojego kodu
    model = MaskedAutoencoderViT(
        img_size=(32, 512),
        patch_size=8,          # Zmienione z 16 na 8 dla detali jidysz
        in_chans=1,            # Twoje zdjęcia BW
        embed_dim=768,         # Parametry dla Basels ./da  
        depth=12,              #
        num_heads=12,          #
        decoder_embed_dim=512, #
        decoder_depth=8,       #
        norm_pix_loss=True     # Krytyczne dla stabilności w OCR
    )
    

    # 3. Przygotowanie danych
    lines_dir = "./data/yiddish_lines"
    dataset = YiddishSharedInRamDataset(lines_dir, img_size=(32, 512))
    # Przy 2x RTX 3090 możesz ustawić duży batch (np. 128 na kartę = 256 łącznie)
    dataloader = DataLoader(dataset,
        batch_size=256,
        shuffle=True,
        num_workers=6, 
        pin_memory=True,      # Przyspiesza transfer RAM -> GPU
        persistent_workers=True, # KLUCZOWE: nie zabija procesów między epokami
        prefetch_factor=4     # Każdy worker przygotowuje 2 batche "na zapas")
    )

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    # 5. Przygotowanie wszystkiego przez accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model = model.to(accelerator.device)
    model = torch.compile(model, mode="reduce-overhead")


    model.train()


    for epoch in range(10):

        for step, batch in enumerate(dataloader):
            batch = batch.to(accelerator.device, non_blocking=True)
            batch = batch.to(torch.bfloat16).div_(255.0)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, _, _ = model(batch, mask_ratio=0.75)
            
            accelerator.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
