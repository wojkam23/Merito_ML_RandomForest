import os
import cv2

# Ustawienia cropowania logotypów – współrzędne przyjęte dla rozdzielczości 1920x1080
# Każdy kanał ma przypisany obszar: (x, y, szerokość, wysokość)
crop_settings = {
    'food':       {'x': 1650, 'y': 900, 'w': 200, 'h': 150},
    'metro':      {'x': 1550, 'y': 30,  'w': 250, 'h': 120},
    'travel':     {'x': 1550, 'y': 833, 'w': 300, 'h': 120},
    'ttv':        {'x': 90,   'y': 70,  'w': 175, 'h': 120},
    'tvn':        {'x': 1620, 'y': 60,  'w': 180, 'h': 130},
    'tvn7':       {'x': 1620, 'y': 60,  'w': 155, 'h': 130},
    'tvn24':      {'x': 30,   'y': 775, 'w': 225, 'h': 175},
    'tvnfabula':  {'x': 75,   'y': 37,  'w': 190, 'h': 140},
    'tvnstyle':   {'x': 1650, 'y': 70,  'w': 200, 'h': 150},
    'tvnturbo':   {'x': 130,  'y': 50,  'w': 310, 'h': 100},
    'warner':     {'x': 1690, 'y': 30,  'w': 150, 'h': 120},
}

# Foldery wejściowy i wyjściowy
input_root = 'logos'        # Folder z pełnymi obrazami (np. klatki z materiałów TV)
output_root = 'logos_crop'  # Folder na zapisane wycinki logotypów

# Iteracja po wszystkich kanałach zdefiniowanych w crop_settings
for channel, settings in crop_settings.items():
    input_dir = os.path.join(input_root, channel)
    output_dir = os.path.join(output_root, channel)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f" Folder {input_dir} nie istnieje. Pomijam.")
        continue

    # Iteracja po wszystkich obrazach w folderze kanału
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_dir, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"Nie można wczytać: {filename}")
                continue

            # Wycinek logotypu zgodnie z ustalonymi współrzędnymi
            x, y, w, h = settings['x'], settings['y'], settings['w'], settings['h']
            crop = img[y:y+h, x:x+w]

            # Zapis wyciętego obrazu do katalogu wyjściowego
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, crop)
            print(f" Wycieto i zapisano: {output_path}")

print("\n Gotowe! Wszystkie logotypy zostały wycięte i zapisane.")
