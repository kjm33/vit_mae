import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class YiddishSharedInRamDataset(Dataset):
    def __init__(self, root_dir, img_size=(32, 512)):
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Sprawdzamy czy ścieżka w ogóle istnieje
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Ścieżka nie istnieje: {os.path.abspath(root_dir)}")

        # POPRAWKA: Rozszerzone filtrowanie (dodany .tiff, .tif i .lower() dla Case-Insensitivity)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
        self.file_names = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        
        # Jeśli lista jest pusta, rzucamy błąd od razu tutaj, zamiast w DataLoaderze
        if len(self.file_names) == 0:
            raise ValueError(
                f"Nie znaleziono obrazów w: {os.path.abspath(root_dir)}. "
                f"Szukane rozszerzenia: {valid_extensions}"
            )
        
        print(f"Ładowanie {len(self.file_names)} obrazów do RAM...")
        
        # Pre-alokacja
        self.data = torch.empty((len(self.file_names), img_size[0], img_size[1]), dtype=torch.uint8)
        
        for idx, name in enumerate(tqdm(self.file_names)):
            img_path = os.path.join(self.root_dir, name)
            try:
                with Image.open(img_path) as img:
                    # Konwersja i resize (W, H dla PIL)
                    img = img.convert('L').resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
                    self.data[idx] = torch.from_numpy(np.array(img, dtype=np.uint8))
            except Exception as e:
                print(f"Błąd przy ładowaniu {name}: {e}")
                # Opcjonalnie: wypełnij pustym obrazem, żeby nie psuć indeksowania
                self.data[idx] = torch.zeros((img_size[0], img_size[1]), dtype=torch.uint8)
        
        self.data = self.data.share_memory_()
        print(f"Dataset gotowy. Załadowano: {self.data.shape[0]} obrazów.")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0)