from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Datos de ejemplo (reemplaza con tus datos reales)
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [1, 0, 1, 0]
})
y = pd.Series([3, 3, 4, 4])  # Etiquetas con más de una clase

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Entrenamiento del modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Lista de todas las etiquetas posibles
all_labels = sorted(y.unique())

# Métricas con todas las etiquetas posibles
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=all_labels))
print("Classification Report:\n", classification_report(y_test, y_pred, labels=all_labels, zero_division=0))

# Visualización del árbol
plt.figure(figsize=(10, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(cls) for cls in all_labels],  # Convertir clases a cadenas
    filled=True
)
plt.show()


