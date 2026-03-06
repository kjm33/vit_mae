import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class YiddishMAEPretrainDataset(Dataset):
    def __init__(self, image_folder, img_size=(32, 512)):
        """
        image_folder: ścieżka do folderu ze skanami linii
        img_size: docelowy rozmiar (H, W)
        """
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        self.h, self.w = img_size

    def __len__(self):
        return len(self.image_paths)

    def prepare_image(self, path):
        # 1. Wczytanie w skali szarości
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None

        # # 2. Resizing z zachowaniem proporcji (Aspect Ratio)
        # h_orig, w_orig = img.shape
        # scale = min(self.h / h_orig, self.w / w_orig)
        # new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        # img_res = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # # 3. Tworzenie białego tła (padding) i centrowanie
        # # W MAE używamy 255 dla białego tła
        # canvas = np.full((self.h, self.w), 255, dtype=np.uint8)
        
        # y_off = (self.h - new_h) // 2
        # x_off = (self.w - new_w) // 2
        # canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img_res

        # 4. Normalizacja do zakresu [0, 1] i zmiana na tensor
        # Model MAE z models_mae.py oczekuje float32
        img_tensor = torch.from_numpy(img).float() / 255.0
        
        # Dodanie wymiaru kanału (1, H, W)
        return img_tensor.unsqueeze(0)

    def __getitem__(self, idx):
        img = self.prepare_image(self.image_paths[idx])
        # Jeśli obraz jest uszkodzony, zwracamy zerowy tensor (lub obsłuż to inaczej)
        if img is None:
            return torch.zeros((1, self.h, self.w))
        return img