import os
import cv2
import numpy as np
import pickle
from randomForest import RandomForest
from sklearn.model_selection import train_test_split

# --- PARAMETRY ---
logos_folder = 'logos_gray_resized'  # Folder z obrazami logotypów (w skali szarości i rozmiarze 50x50)
resize_size = (50, 50)               # Docelowy rozmiar obrazów (dla bezpieczeństwa zgodności rozmiaru)

# --- Przygotowanie danych ---
X = []  # Lista cech (spłaszczone obrazy)
y = []  # Lista etykiet (nazwa kanału)

print("Przygotowywanie danych...")

# Iteracja po folderach reprezentujących poszczególne klasy/kanały TV
for channel_name in os.listdir(logos_folder):
    channel_path = os.path.join(logos_folder, channel_name)

    if not os.path.isdir(channel_path):
        continue  # Pomija pliki – tylko foldery nas interesują

    # Iteracja po plikach graficznych danego kanału
    for filename in os.listdir(channel_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(channel_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Wczytanie jako obraz grayscale
            if img is None:
                continue
            img_resized = cv2.resize(img, resize_size)  # Dla pewności wymuszamy rozmiar
            flat = img_resized.flatten()  # Spłaszczamy obraz do wektora 1D
            X.append(flat)
            y.append(channel_name)  # Etykieta = nazwa folderu/kanału

# Konwersja do tablic numpy
X = np.array(X)
y = np.array(y)

print(f"Dane przygotowane: {X.shape[0]} próbek, {X.shape[1]} cech każda.")

# --- Podział na zbiór treningowy i testowy ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Trenowanie własnego modelu Random Forest ---
print("Trenowanie Random Forest...")
forest = RandomForest(n_estimators=200, max_depth=50)  # 200 drzew, każde max głębokość 50
forest.fit(X_train, y_train)

# --- Ocena skuteczności ---
y_pred = forest.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nDokładność na zbiorze testowym: {accuracy * 100:.2f}%")

# --- Zapis wytrenowanego modelu do pliku ---
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

print("Model zapisany jako random_forest_model.pkl!")
