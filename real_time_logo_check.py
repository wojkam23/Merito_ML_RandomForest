import cv2
import numpy as np
import pickle
import os
from collections import Counter, deque

# --- Wczytanie wytrenowanego modelu Random Forest ---
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Współrzędne cropowania logotypów dla obrazu 1920x1080 ---
# Stałe współrzędne, ręcznie dobrane per kanał (x, y, szerokość, wysokość)
crop_params = {
    'food': (1650, 900, 200, 150),
    'metro': (1550, 30, 250, 120),
    'travel': (1550, 833, 300, 120),
    'ttv': (90, 70, 175, 120),
    'tvn': (1620, 60, 180, 130),
    'tvn7': (1620, 60, 155, 130),
    'tvn24': (30, 775, 225, 175),
    'tvnfabula': (75, 37, 190, 140),
    'tvnstyle': (1650, 70, 200, 150),
    'tvnturbo': (130, 50, 310, 100),
    'warner': (1690, 30, 150, 120),
}

resize_size = (50, 50)  # rozmiar logotypu po przeskalowaniu

# --- Funkcje pomocnicze ---
def resize_frame(frame):
    # Dopasowanie do standardowego rozmiaru
    return cv2.resize(frame, (1920, 1080))

def preprocess_logo(logo_crop):
    # Przekształcenie obrazu logo: grayscale → resize → spłaszczenie do wektora
    gray = cv2.cvtColor(logo_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, resize_size)
    flat = resized.flatten()
    return flat.reshape(1, -1)  # kształt (1, 2500)

# ---  Parametry decyzyjne ---
okno_predykcji = 5           # Ile ostatnich sekund uwzględnić w decyzji
prog_pewnosci_logo = 65      # Pewność (%) powyżej której uznajemy logo za wiarygodne
prog_pewnosci_niskie = 50    # Minimalna pewność, by rozważać logo jako możliwe

# --- Wczytanie pliku wideo ---
video_folder = 'video'
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
if not video_files:
    print("Brak plików .mp4 w folderze 'video'!")
    exit(1)

print("\nDostępne pliki video:")
for idx, file in enumerate(video_files):
    print(f"[{idx}] {file}")

file_idx = int(input("\nWybierz numer pliku: "))
if file_idx < 0 or file_idx >= len(video_files):
    print("Błędny numer pliku!")
    exit(1)

selected_video = os.path.join(video_folder, video_files[file_idx])
print(f"Wybrano plik: {selected_video}")

cap = cv2.VideoCapture(selected_video)
if not cap.isOpened():
    print(f"Nie można otworzyć pliku: {selected_video}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # analizujemy 1 klatkę na sekundę
frame_count = 0

# Bufor do uśredniania predykcji z ostatnich sekund
predictions_buffer = deque(maxlen=okno_predykcji)

# --- Główna pętla przetwarzania klatek ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_frame(frame)

    if frame_count % frame_interval == 0:
        pewnosci = {}  # kanał → (etykieta_predykcji, pewność)

        # Dla każdego kanału wytnij logotyp i dokonaj predykcji
        for channel, (x, y, w, h) in crop_params.items():
            logo_crop = frame[y:y+h, x:x+w]
            try:
                features = preprocess_logo(logo_crop)
                # Zbierz predykcje wszystkich drzew
                tree_preds = np.array([tree.predict(features) for tree in model.trees])
                votes = Counter(tree_preds.flatten())
                predicted_label, count = votes.most_common(1)[0]
                total_votes = sum(votes.values())
                confidence = (count / total_votes) * 100
                pewnosci[channel] = (predicted_label, confidence)
            except Exception:
                pewnosci[channel] = ("error", 0)

        # Wybierz kanał z największą pewnością
        najlepszy_kanal = max(pewnosci.items(), key=lambda item: item[1][1])
        najlepszy_nazwa, (najlepszy_label, najlepszy_pewnosc) = najlepszy_kanal

        predictions_buffer.append((najlepszy_label, najlepszy_pewnosc))

        # Gdy bufor pełny, podejmij decyzję końcową
        if len(predictions_buffer) == okno_predykcji:
            wszystkie_labelki = [x[0] for x in predictions_buffer]
            najczestszy_label, ile_razy = Counter(wszystkie_labelki).most_common(1)[0]
            pewnosci_najczestszego = [
                pewnosc for label, pewnosc in predictions_buffer if label == najczestszy_label
            ]
            srednia_pewnosc = np.mean(pewnosci_najczestszego)

            # --- Decyzja ---
            if srednia_pewnosc >= prog_pewnosci_logo:
                print(f"✅ {najczestszy_label.upper()} --> logo prawidłowe! (średnia pewność: {srednia_pewnosc:.1f}%)")
            elif prog_pewnosci_niskie <= srednia_pewnosc < prog_pewnosci_logo:
                print(f"⚠️ Niska pewność: możliwe logo {najczestszy_label.upper()} (średnia pewność: {srednia_pewnosc:.1f}%)")
            else:
                print(f"🎥 Reklama! (średnia pewność: {srednia_pewnosc:.1f}%)")

    # Wyświetl klatkę
    cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC - zakończenie programu
        break

    frame_count += 1

#  Zakończenie
cap.release()
cv2.destroyAllWindows()
