# Wykrywanie Logotypów Kanałów TV przy użyciu Random Forest



Celem projektu jest **automatyczna klasyfikacja logotypów telewizyjnych** na podstawie obrazów wyciętych z materiałów wideo. Klasyfikator został zaimplementowany ręcznie w języku Python przy użyciu algorytmu **Random Forest**, bez wykorzystania gotowych bibliotek klasyfikacyjnych typu `sklearn`.

---
## Zdjęcie prezentujące działanie
![image](https://github.com/user-attachments/assets/2cbb926a-9a82-4fa8-91d7-0f8ec791f38f)

---

## Struktura katalogów
├── logos/ # Surowe obrazy z materiałów wideo (jeden folder na kanał)

├── logos_crop/ # Wycięte logotypy z obrazów (na podstawie współrzędnych)

├── logos_gray/ # Wersje logotypów w skali szarości

├── logos_gray_resized/ # Wersje grayscale + resize (50x50)

├── video/ # Folder z filmami testowymi (format .mp4)

├── random_forest_model.pkl # Zapisany model Random Forest po trenowaniu

├── X.npy / y.npy # Dane treningowe i etykiety (numpy arrays)


---

## Jak to działa?

### 1. Wycinanie logotypów – `cropLogo.py`
Skrypt wycina fragmenty obrazu, w których znajdują się logotypy kanałów. Działa na podstawie ręcznie zdefiniowanych współrzędnych przycięcia (`x`, `y`, `szerokość`, `wysokość`) dostosowanych do rozdzielczości 1920x1080.

---

### 2. Przetwarzanie obrazów – `greyScale.py`
Konwertuje logotypy do:
- skali szarości (`logos_gray/`)
- wersji 50x50 px (`logos_gray_resized/`)

Następnie spłaszcza obrazy i zapisuje je jako:
- `X.npy` — dane wejściowe (cechy: 2500 na próbkę)
- `y.npy` — etykiety tekstowe (np. "tvn", "metro")

---

### 3. Trening klasyfikatora – `main.py`
Uczenie modelu Random Forest:
- Liczba drzew: 200
- Głębokość: 50
- Podział danych: 80% trening / 20% test

Model zapisuje się jako `random_forest_model.pkl`, a wynik działania wyświetla dokładność klasyfikacji.

---

### 4. Klasyfikator Random Forest – `randomForest.py`
Ręcznie zaimplementowana klasa `RandomForest` oraz `DecisionTree`, bez użycia gotowych modeli z bibliotek. Funkcjonalności:
- Bootstrap sampling
- Losowy wybór cech i podziałów
- Majority voting
- Metody diagnostyczne (głębokość drzewa, podziały, użyte cechy)

---

### 5. Test na pliku wideo – `real_time_logo_check.py`
Skrypt:
- Pozwala wybrać plik `.mp4` z folderu `video/`
- Przetwarza wideo klatka po klatce
- Wycina logotyp z każdej klatki i klasyfikuje go
- Stosuje bufor predykcji z kilku sekund
- Określa z wysoką/średnią/małą pewnością, który kanał jest wyświetlany – lub że to reklama

---

## Przykładowy wynik
✅ TVN --> logo prawidłowe! (średnia pewność: 84.7%)

⚠️ Niska pewność: możliwe logo METRO (średnia pewność: 55.2%)

🎥 Reklama! (średnia pewność: 43.9%)

---

## Bibliografia
- Colab Codes: "Implementing Random Forests in Python on Iris Dataset"

https://www.colabcodes.com/post/implementing-random-forests-in-python-on-iris-dataset
- Medium Article: "Random Forest Classifier from Scratch"

https://medium.com/@poleakchanrith/random-forest-classifier-implementation-from-scratch-with-python-8cee705624e7
- Machine Learning Mastery: "Random Forest from Scratch in Python" 

https://machinelearningmastery.com/implement-random-forest-scratch-python/
- Kaggle: "Random Forest From Scratch"

https://www.kaggle.com/code/fareselmenshawii/random-forest-from-scratch
