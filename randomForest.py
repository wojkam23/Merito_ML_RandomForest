import numpy as np
from collections import Counter

# --------------------
# Klasa DecisionTree – pojedyncze drzewo decyzyjne
# --------------------
class DecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth              # Maksymalna głębokość drzewa
        self.tree = None                        # Zmienna przechowująca strukturę drzewa
        self.feature_splits = []                # Lista cech i progów użytych do podziałów (do diagnostyki)

    def fit(self, X, y):
        # Buduje drzewo rekurencyjnie
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        # Predykcja dla wielu próbek – każda przechodzi przez drzewo
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        # Warunki zakończenia:
        # - osiągnięto maksymalną głębokość
        # - wszystkie etykiety są takie same
        if depth == self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]  # Zwróć najczęściej występującą etykietę (liść)

        # Wybierz losowo jedną cechę do podziału
        feature = np.random.randint(0, X.shape[1])
        threshold = float(np.median(X[:, feature]))  # Próg = mediana wartości tej cechy

        self.feature_splits.append((feature, threshold))

        # Podział danych względem progu
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        # Jeśli podział nie ma sensu – zakończ i zwróć najczęstszą klasę
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return Counter(y).most_common(1)[0][0]

        # Rekurencyjne budowanie poddrzew
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Struktura: (cecha, próg, lewe_poddrzewo, prawe_poddrzewo)
        return (feature, threshold, left_subtree, right_subtree)

    def _traverse_tree(self, x, node):
        # Przechodzenie po drzewie dla jednej próbki x
        if not isinstance(node, tuple):  # Liść – zwróć etykietę
            return node
        feature, threshold, left_subtree, right_subtree = node
        if x[feature] <= threshold:
            return self._traverse_tree(x, left_subtree)
        else:
            return self._traverse_tree(x, right_subtree)

    def get_tree_info(self):
        # Zwraca diagnostyczne informacje o drzewie
        return {
            "depth": self.max_depth,
            "splits": len(self.feature_splits),
            "features_used": self.feature_splits
        }

# --------------------
# Klasa RandomForest – las losowy złożony z wielu drzew
# --------------------
class RandomForest:
    def __init__(self, n_estimators=5, max_depth=2):
        self.n_estimators = n_estimators      # Liczba drzew
        self.max_depth = max_depth            # Maksymalna głębokość pojedynczego drzewa
        self.trees = []                       # Lista drzew w lesie

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap – losujemy podzbiór danych ze zwracaniem
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Trenuj nowe drzewo
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Predykcje wszystkich drzew → głosowanie większościowe
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_trees, n_samples)

        # Dla każdej próbki bierzemy najczęściej występującą etykietę
        return [
            Counter(tree_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ]

    def _bootstrap_sample(self, X, y):
        # Tworzy próbkę bootstrapową (z powtórzeniami)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples // 2, replace=True)
        return X[indices], y[indices]

    def display_forest_info(self):
        # Diagnostyka lasu – wypisz parametry każdego drzewa
        print("Informacje o drzewach w lesie:")
        for i, tree in enumerate(self.trees):
            tree_info = tree.get_tree_info()
            print(f"Drzewo {i + 1}:")
            print(f" - Głębokość: {tree_info['depth']}")
            print(f" - Liczba podziałów: {tree_info['splits']}")
            print(f" - Użyte cechy i progi podziałów: {tree_info['features_used']}")
            print("-" * 40)
