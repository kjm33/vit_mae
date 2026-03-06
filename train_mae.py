import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from models_mae import MaskedAutoencoderViT # Import z Twojego pliku
from yiddish_mare_pretrain_ds import YiddishMAEPretrainDataset # Wcześniej przygotowany dataloader

def train():
    # 1. Inicjalizacja Accelerate z wymuszeniem mixed_precision="bf16"
    accelerator = Accelerator(mixed_precision="bf16")
    
    # 2. Konfiguracja modelu pod parametry 32x512 i 1 kanał
    # Wykorzystujemy architekturę ViT-Base z Twojego kodu
    model = MaskedAutoencoderViT(
        img_size=(32, 512),
        patch_size=8,          # Zmienione z 16 na 8 dla detali jidysz
        in_chans=1,            # Twoje zdjęcia BW
        embed_dim=768,         # Parametry dla Base
        depth=12,              #
        num_heads=12,          #
        decoder_embed_dim=512, #
        decoder_depth=8,       #
        norm_pix_loss=True     # Krytyczne dla stabilności w OCR
    )

    # 3. Przygotowanie danych
    dataset = YiddishMAEPretrainDataset("./data/yiddish_lines", img_size=(32, 512))
    # Przy 2x RTX 3090 możesz ustawić duży batch (np. 128 na kartę = 256 łącznie)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

    # 5. Przygotowanie wszystkiego przez accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(200):
        for step, batch in enumerate(dataloader):
            # Model MAE przyjmuje obrazy i zwraca (loss, pred, mask)
            # Domyślny mask_ratio to 0.75
            loss, _, _ = model(batch, mask_ratio=0.75)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # Zapisywanie modelu co epokę (tylko na głównym procesie)
        if accelerator.is_main_process:
            accelerator.save_state("mae_checkpoint_yiddish")

if __name__ == "__main__":
    train()