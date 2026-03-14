import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class YiddishSharedInRamDataset(Dataset):
    def __init__(self, root_dir, img_size=(32, 512)):
        self.root_dir = root_dir
        self.img_size = img_size # (H, W) -> (32, 512)
        
        # Pobieramy listę plików
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Ładowanie {len(self.file_names)} obrazów do RAM...")
        
        # Pre-alokacja tensora w RAM (Batch, H, W) - oszczędza czas stackowania
        self.data = torch.empty((len(self.file_names), img_size[0], img_size[1]), dtype=torch.uint8)
        
        for idx, name in enumerate(tqdm(self.file_names)):
            img_path = os.path.join(self.root_dir, name)
            
            # Otwieramy, konwertujemy na skalę szarości i zmieniamy rozmiar
            with Image.open(img_path) as img:
                img = img.convert('L').resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
                # PIL resize bierze (W, H), stąd zamiana kolejności powyżej
                
                # Zapisujemy jako uint8 (0-255)
                self.data[idx] = torch.from_numpy(np.array(img, dtype=np.uint8))
        
        # KLUCZOWE: Przenosimy do Shared Memory dla multiprocessing
        self.data = self.data.share_memory_()
        print("Dataset gotowy w pamięci współdzielonej.")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        # Pobieramy surowy obraz uint8
        img = self.data[idx] 
        
        # Zwracamy z wymiarem kanału (1, 32, 512)
        # Nie robimy tu float() ani /255 - to zrobi GPU w pętli treningowej
        return img.unsqueeze(0)