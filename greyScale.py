import os
import cv2
import numpy as np

# --- Ścieżki wejścia/wyjścia ---
input_root = 'logos_crop'                   # Folder z kolorowymi logotypami
output_gray_root = 'logos_gray'             # Folder do zapisu wersji grayscale
output_gray_resized_root = 'logos_gray_resized'  # Folder do zapisu 50x50 grayscale
os.makedirs(output_gray_root, exist_ok=True)
os.makedirs(output_gray_resized_root, exist_ok=True)

X = []  # cechy (obrazy spłaszczone do wektora)
y = []  # etykiety (nazwy kanałów)

resize_size = (50, 50)  # Rozmiar do którego skalujemy logotypy

# --- Iteracja po folderach (klasach/kanałach) ---
for channel in os.listdir(input_root):
    channel_path = os.path.join(input_root, channel)
    if not os.path.isdir(channel_path):
        continue  # Pomija pliki

    # Utwórz foldery docelowe dla grayscale i grayscale+resize
    gray_channel_path = os.path.join(output_gray_root, channel)
    gray_resized_channel_path = os.path.join(output_gray_resized_root, channel)
    os.makedirs(gray_channel_path, exist_ok=True)
    os.makedirs(gray_resized_channel_path, exist_ok=True)

    # Przejdź po wszystkich obrazach danego kanału
    for filename in os.listdir(channel_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(channel_path, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"Nie można wczytać: {filepath}")
                continue

            # Konwersja do skali szarości
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Zapisz grayscale
            gray_save_path = os.path.join(gray_channel_path, filename)
            cv2.imwrite(gray_save_path, gray)

            # Skalowanie do 50x50
            gray_resized = cv2.resize(gray, resize_size)

            # Zapisz skalowany grayscale
            gray_resized_save_path = os.path.join(gray_resized_channel_path, filename)
            cv2.imwrite(gray_resized_save_path, gray_resized)

            # Przygotuj dane do treningu (spłaszczone)
            X.append(gray_resized.flatten())  # 50x50 → 2500 cech
            y.append(channel)

# --- Zapis danych do plików NumPy ---
X = np.array(X)
y = np.array(y)

np.save('X.npy', X)
np.save('y.npy', y)

# Podsumowanie
print("\nGotowe! Wszystko zapisane:")
print("- Grayscale w logos_gray/")
print("- Grayscale + Resize w logos_gray_resized/")
print("- Cechy w pliku X.npy")
print("- Etykiety w pliku y.npy")
