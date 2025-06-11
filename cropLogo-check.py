import cv2
import os

# Ścieżki do folderów z obrazami (po jednym dla każdego kanału)
input_folders = {
    'food': 'logos/food',
    'metro': 'logos/metro',
    'travel': 'logos/travel',
    'ttv': 'logos/ttv',
    'tvn': 'logos/tvn',
    'tvn7': 'logos/tvn7',
    'tvn24': 'logos/tvn24',
    'tvnfabula': 'logos/tvnfabula',
    'tvnstyle': 'logos/tvnstyle',
    'tvnturbo': 'logos/tvnturbo',
    'warner': 'logos/warner',
}

# Folder na zapisane wycinki (tylko po jednym na kanał)
output_folder = 'logos_cropped_test'
os.makedirs(output_folder, exist_ok=True)

#  Współrzędne cropowania – jak w głównym pipeline
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

# Przetwarzanie – tylko jeden plik z każdego kanału
for channel, folder in input_folders.items():
    try:
        # Wczytaj pierwszy plik z folderu
        filename = sorted(os.listdir(folder))[0]
        filepath = os.path.join(folder, filename)

        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Nie można wczytać obrazu.")

        # Pobierz parametry cropowania
        x, y, w, h = crop_params[channel]
        cropped = img[y:y+h, x:x+w]

        # Zapisz wycinek jako testowy plik
        output_path = os.path.join(output_folder, f'{channel}_crop.jpg')
        cv2.imwrite(output_path, cropped)
        print(f'Wycięte {channel} do {output_path}')

    except Exception as e:
        print(f'Błąd dla {channel}: {e}')
