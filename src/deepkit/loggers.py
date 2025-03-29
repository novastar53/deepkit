import os
import random
import string
import numpy as np
from pathlib import Path

__all__ = ["DiskLogger"]

def generate_random_code(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

class DiskLogger:
    def __init__(self, name, save_dir="./logs"):
        self.run_code = generate_random_code()
        self.save_dir = Path(save_dir) / self.run_code / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.files = None

    def log(self, step, x):
        file_name = f"{step:05d}.npz"
        file_path = self.save_dir / file_name
        np.savez(file_path, **x)

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __call__(self):
        if not self.files:
          self.files = sorted([f for f in os.listdir(self.save_dir)])
        for filename in self.files:
          filepath = os.path.join(self.save_dir, filename)
          data = np.load(filepath, allow_pickle=True)
          yield data

    def __getitem__(self, i):
       if not self.files:
        self.files = sorted([f for f in os.listdir(self.save_dir)])
       filepath = os.path.join(self.save_dir, self.files[i])
       data = np.load(filepath, allow_pickle=True)
       return data

