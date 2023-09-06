"""
Archivo principal

Aquí se ejecuta el programa principal.

Entregable: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.
"""

__author__ = "Eduardo Joel Cortez Valente"

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

data = load_digits()
X = data.data
y = data.target

# Divide los datos en conjuntos de entrenamiento y prueba
X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
Parámetros para el modelo y su justificación:
    
    - El hiperparámetro random_state se utiliza para controlar la aleatoriedad en la construcción del árbol.
    Al fijar este valor en 32, estás garantizando que el proceso de división y construcción del árbol sea reproducible.
    
    - El hiperparámetro min_samples_split controla el número mínimo de muestras requeridas para que un nodo se divida 
    en subnodos. Al establecerlo en 2, estás permitiendo que el árbol se divida incluso si solo hay dos muestras en un 
    nodo. Esta elección puede hacer que el árbol sea más profundo y más flexible para ajustarse a los datos, pero 
    también puede aumentar el riesgo de sobreajuste.
    
    - El hiperparámetro max_depth controla la profundidad máxima del árbol. Al establecerlo en None, permites que el 
    árbol crezca hasta que todas las hojas sean puras o hasta que se alcance el min_samples_split mínimo. Esto significa 
    que el árbol puede crecer hasta su máxima profundidad, lo que puede resultar en un árbol más complejo y posiblemente 
    sobreajustado.
"""

random_state = 32
samples_split = 2
depth = None

# Divide el conjunto de entrenamiento en entrenamiento y validación
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, test_size=0.2, random_state=0)

# Crea un modelo de árbol de decisión
classifier = DecisionTreeClassifier(min_samples_split=samples_split, max_depth=depth)

# Entrena el modelo en el conjunto de entrenamiento
classifier.fit(X_train, Y_train)

# Realiza predicciones en el conjunto de validación
Y_pred_validation = classifier.predict(X_validation)
score_validation = accuracy_score(Y_validation, Y_pred_validation)
conf_matrix_validation = confusion_matrix(Y_validation, Y_pred_validation)
recall_validation = recall_score(Y_validation, Y_pred_validation, average='macro')

# Ajusta el modelo nuevamente con el conjunto de entrenamiento completo
classifier.fit(X_train_initial, Y_train_initial)

# Realiza predicciones en el conjunto de prueba
Y_pred = classifier.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
conf_matrix_test = confusion_matrix(Y_test, Y_pred)
recall_test = recall_score(Y_test, Y_pred, average='macro')

# Graficación de los resultados
labels = ['Validación', 'Prueba']
accuracy_scores = [score_validation, score]
recall_scores = [recall_validation, recall_test]

# Gráfico de barras para precisión
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, accuracy_scores, color=['green', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Precisión')
plt.title('Precisión en Validación y Prueba')

# Gráfico de barras para recall
plt.subplot(1, 2, 2)
plt.bar(labels, recall_scores, color=['green', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Recall')
plt.title('Recall en Validación y Prueba')

plt.tight_layout()
plt.show()

# Matriz de Confusión en el conjunto de validación
print("Matriz de Confusión en el conjunto de validación:")
print(conf_matrix_validation)

# Matriz de Confusión en el conjunto de prueba
print("Matriz de Confusión en el conjunto de prueba:")
print(conf_matrix_test)

# Gráfico de Matriz de Confusión en el conjunto de validación
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_validation, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión en Validación')
plt.show()

# Gráfico de Matriz de Confusión en el conjunto de prueba
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión en Prueba')
plt.show()
