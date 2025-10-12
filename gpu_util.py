import torch
import pandas as pd

# Perintah ini akan mengembalikan True jika PyTorch bisa mendeteksi GPU yang kompatibel
print(torch.cuda.is_available())