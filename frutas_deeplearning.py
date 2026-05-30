import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

manzana = '/content/drive/MyDrive/Patrones/manzanas'

# Obtener lista de archivos en la carpeta
train_manzana= [os.path.join(manzana, file) for file in os.listdir(manzana)]

# Mostrar las rutas de los archivos
for path in train_manzana:
    print(path)


platano = '/content/drive/MyDrive/Patrones/platano'

# Obtener lista de archivos en la carpeta
train_platano= [os.path.join(platano, file) for file in os.listdir(platano)]

# Mostrar las rutas de los archivos
for path in train_platano:
    print(path)

tunas = '/content/drive/MyDrive/Patrones/tunas'

# Obtener lista de archivos en la carpeta
train_tuna= [os.path.join(tunas, file) for file in os.listdir(tunas)]

# Mostrar las rutas de los archivos
for path in train_tuna:
    print(path)

# nada = '/content/drive/MyDrive/Patrones/nada'

# # Obtener lista de archivos en la carpeta
# train_nada= [os.path.join(nada, file) for file in os.listdir(nada)]

# # Mostrar las rutas de los archivos
# for path in train_nada:
#     print(path)
from google.colab import drive
drive.mount('/content/drive')

def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convertir a escala de grises
    return np.array(img)

def preprocess_image(image, target_size=(64, 64)):
    image = Image.fromarray(image).resize(target_size)
    return np.array(image)

def normalize_image(image):
    return image / 255.0

X_train = []
Y_train = []

# Itera sobre todas las imágenes de la carpeta de manzana
for image_path in train_manzana:
    image = load_image(image_path)         # Cargar la imagen
    image = preprocess_image(image)        # Redimensionar la imagen
    image = normalize_image(image)         # Normalizar la imagen
    X_train.append(image)                  # Agregar imagen preprocesada a la lista de datos de entrenamiento
    Y_train.append(0)                      # Etiqueta para clase glioma

# Itera sobre todas las imágenes de la carpeta de platano
for image_path in train_platano:
    image = load_image(image_path)         # Cargar la imagen
    image = preprocess_image(image)        # Redimensionar la imagen
    image = normalize_image(image)         # Normalizar la imagen
    X_train.append(image)                  # Agregar imagen preprocesada a la lista de datos de entrenamiento
    Y_train.append(1)                      # Etiqueta para clase no tumor

# Itera sobre todas las imágenes de la carpeta de tunas
for image_path in train_tuna:
    image = load_image(image_path)         # Cargar la imagen
    image = preprocess_image(image)        # Redimensionar la imagen
    image = normalize_image(image)         # Normalizar la imagen
    X_train.append(image)                  # Agregar imagen preprocesada a la lista de datos de entrenamiento
    Y_train.append(2)                      # Etiqueta para clase no tumor

print(len(X_train))
print(len(Y_train))
print(Y_train)

import numpy as np

# Kernels para detección de bordes
kernel1 = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1],
                   ])

kernel2 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

kernel_diagonal1 = np.array([[1, 0, -1],
                             [0, 1, 0],
                             [-1, 0, 1]])

kernel_diagonal2 = np.array([[-1, 0, 1],
                             [0, 1, 0],
                             [1, 0, -1]])

def convolucion(imagen, kernel):
    imagen_h, imagen_w = imagen.shape
    kernel_h, kernel_w = kernel.shape
    new_image_h = imagen_h - kernel_h + 1
    new_image_w = imagen_w - kernel_w + 1
    nueva_imagen = np.zeros((new_image_h, new_image_w))
    for y in range(new_image_h):
        for x in range(new_image_w):
            nueva_imagen[y, x] = np.sum(imagen[y:y+kernel_h, x:x+kernel_w] * kernel)
    return nueva_imagen

def relu(x):
    return np.maximum(0, x)

def max_pooling(nueva_imagen, pool_size):
    imagen_h, imagen_w = nueva_imagen.shape
    new_image_h = imagen_h - pool_size + 1
    new_image_w = imagen_w - pool_size + 1
    new_image = np.zeros((new_image_h, new_image_w))
    for y in range(new_image_h):
        for x in range(new_image_w):
            new_image[y, x] = np.max(nueva_imagen[y:y+pool_size, x:x+pool_size])
    return new_image

def aplicar_capa(imagen, kernel1, kernel2, kernel_diagonal1, kernel_diagonal2, pool_size):
    # Aplicación de los kernels
    conv1 = convolucion(imagen, kernel1)
    relu1 = relu(conv1)
    pool1 = max_pooling(relu1, pool_size)

    conv2 = convolucion(pool1, kernel2)
    relu2 = relu(conv2)
    pool2 = max_pooling(relu2, pool_size)

    conv_diagonal1 = convolucion(pool2, kernel_diagonal1)
    relu_diagonal1 = relu(conv_diagonal1)
    pool_diagonal1 = max_pooling(relu_diagonal1, pool_size)

    conv_diagonal2 = convolucion(pool_diagonal1, kernel_diagonal2)
    relu_diagonal2 = relu(conv_diagonal2)
    pool_diagonal2 = max_pooling(relu_diagonal2, pool_size)

    return pool_diagonal2

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(X_train.shape)


import matplotlib.pyplot as plt


X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Parámetros
num_capas = 2
pool_size = 2

# Aplicar las capas de convolución, ReLU y max pooling
resultados = np.array(X_train)
for _ in range(num_capas):
    print(f"Capa no. {_ + 1} iniciada")
    resultados = np.array([aplicar_capa(imagen, kernel1, kernel2, kernel_diagonal1, kernel_diagonal2, pool_size) for imagen in resultados])
    print(f"Capa no. {_ + 1} finalizada")

# Comprobar el tamaño de cada uno de los resultados finales
for i, resultado in enumerate(resultados):
    print(f"El tamaño del resultado final {i+1} es: {resultado.shape}")

# Visualizar imágenes originales y resultados
plt.figure(figsize=(12, 8))

# Imágenes originales
for i in range(min(4, len(X_train))):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_train[i].reshape(64, 64), cmap='gray')
    plt.title(f'Original {i+1}')
    plt.axis('off')

# Resultados después de la primera capa
for i in range(min(4, len(resultados))):
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(resultados[i].reshape(resultados[0].shape[0], resultados[0].shape[1]), cmap='gray')
    plt.title(f'Resultado {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Visualizar los resultados finales
def visualizar_resultados(resultados):
    num_resultados = len(resultados)
    num_cols = 4  # Número de columnas en la visualización
    num_rows = (num_resultados + num_cols - 1) // num_cols  # Número de filas necesario

    plt.figure(figsize=(15, num_rows * 5))

    for i, resultado in enumerate(resultados):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(resultado, cmap='gray')
        plt.title(f'Resultado {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Llamar a la función de visualización
visualizar_resultados(resultados)

resultados = np.array(resultados)
print(resultados.shape)

import numpy as np

# Resultados de la CNN después de aplicar las capas de convolución y pooling
resultados = np.array(resultados)

# Aplanar las características extraídas por la CNN
X_train_flat = resultados.reshape(resultados.shape[0], -1)

# Comprobar las dimensiones de las características aplanadas
#print(f"Las características aplanadas tienen forma: {X_train_flat.shape}")

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
delta = 0.01
X0 = 1
ECM = 10
epocas = 2000
epoca = 0
alpha = 0.2

# Asumiendo que X es tu conjunto de datos de entrada
# X = ...

# Etiquetas
Yd = np.array([[1,0,0]] * 25 + [[0,1,0]] * 25 + [[0,0,1]] * 25)

np.random.seed(40)
neuronas_entrada = 120
neuronas_ocultas = 60
neuronas_salida = 3  # Cambiado a 3 para las tres clases de frutas

# Capa de entrada
W_entrada = 2*np.random.rand(X.shape[1], neuronas_entrada) - 1

# Capa oculta
W_oculta = 2*np.random.rand(neuronas_entrada+1, neuronas_ocultas) - 1

# Capa salida
W_salida = 2*np.random.rand(neuronas_ocultas+1, neuronas_salida) - 1

# Inicialización de pesos
W_entrada_new = np.zeros(W_entrada.shape)
W_entrada_old = np.zeros(W_entrada.shape)

W_oculta_new = np.zeros(W_oculta.shape)
W_oculta_old = np.zeros(W_oculta.shape)

W_salida_new = np.zeros(W_salida.shape)
W_salida_old = np.zeros(W_salida.shape)

# Funciones de activación y sus derivadas
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def deriv_sigmoide(z):
    return np.exp(-z)/(1.0+np.exp(-z))**2

def tang_hiper(z):
    return np.tanh(z)

def deriv_tang_hiper(z):
    return 1 - np.tanh(z)**2

ECM_camb = []
while (ECM >= 0.001 and epoca <= epocas):
    # Propagación hacia adelante
    z_entrada = np.dot(X, W_entrada)
    y_entrada = np.insert(tang_hiper(z_entrada), 0, -1, axis=1)

    z_oculta = np.dot(y_entrada, W_oculta)
    y_oculta = np.insert(sigmoide(z_oculta), 0, -1, axis=1)

    z_salida = np.dot(y_oculta, W_salida)
    Yobt = sigmoide(z_salida)  # Cambiado a sigmoide para salida multiclase

    # Retropropagación
    aux3 = (Yd - Yobt) * deriv_sigmoide(z_salida)
    aux2 = np.dot(aux3, W_salida[1:,:].T) * deriv_sigmoide(z_oculta)
    aux = np.dot(aux2, W_oculta[1:,:].T) * deriv_tang_hiper(z_entrada)

    # Actualización de pesos con Momentum
    W_salida_new = W_salida + delta * np.dot(y_oculta.T, aux3) + alpha * (W_salida - W_salida_old)
    W_salida_old = W_salida
    W_salida = W_salida_new

    W_oculta_new = W_oculta + delta * np.dot(y_entrada.T, aux2) + alpha * (W_oculta - W_oculta_old)
    W_oculta_old = W_oculta
    W_oculta = W_oculta_new

    W_entrada_new = W_entrada + delta * np.dot(X.T, aux) + alpha * (W_entrada - W_entrada_old)
    W_entrada_old = W_entrada
    W_entrada = W_entrada_new

    # Cálculo del Error Cuadrático Medio
    ECM = np.mean((Yd - Yobt)**2)
    ECM_camb.append(ECM)
    epoca += 1

    if epoca % 100 == 0:  # Imprimir cada 100 épocas para reducir la salida
        print(f"Epoca: {epoca}, ECM: {ECM}")

print("Y obt final:")
print(np.round(Yobt[:5], 3))  # Mostrar solo las primeras 5 predicciones

plt.plot(range(1, epoca + 1), ECM_camb)
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio (ECM)')
plt.title('ECM por época')
plt.grid(True)
plt.show()

# Suponiendo que resultados tiene forma [n_samples, height, width, channels]
X_train_flat = resultados.reshape(resultados.shape[0], -1)
print(f"X_train_flat shape: {X_train_flat.shape}")
# Parámetros de la red ART
n = X_train_flat.shape[1]  # Número de características
m = 3  # Número inicial de neuronas en la capa de salida

# Inicialización de los pesos
pesos_f = np.ones((m, n)) * (1 / (1 + n))
pesos_b = np.ones((n, m)) 
# Parámetros de vigilancia
p = 0.1
limite = 10

# Fase de entrenamiento
for i in range(X_train_flat.shape[0]):
    sample = X_train_flat[i]

    # Calcular el valor de activación
    k = []
    for j in range(pesos_f.shape[0]):
        u_sum_f = np.dot(sample, pesos_f[j])
        k.append((u_sum_f, j))

    # Ordenar valores de activación de mayor a menor
    k_g = [index for _, index in sorted(k, reverse=True)]
    print(f"Sample no. {i} : ")

    # Fase de comparación
    flag = False
    for idx in k_g:
        u_sum_b = np.dot(sample, pesos_b[:, idx])
        sum_F1 = np.sum(sample)
        R = u_sum_b / sum_F1
        if R >= p:
            flag = True
            ganadora = idx
            break

    # Si no se encuentra una coincidencia, se añade una nueva neurona
    if not flag:
        ganadora = pesos_f.shape[0]
        if ganadora < limite:
            Ncol_f = sample.copy()
            Ncol_b = np.ones(sample.size)

            pesos_f = np.vstack([pesos_f, Ncol_f])
            pesos_b = np.column_stack([pesos_b, Ncol_b])

            print(f"Nueva neurona añadida. Total clases: {pesos_f.shape[0]}")

    # Actualización de pesos
    pesos_b[:, ganadora] = pesos_b[:, ganadora] * sample
    pesos_f[ganadora] = pesos_b[:, ganadora] / (0.5 + np.sum(pesos_b[:, ganadora]))

    print(f"Muestra clasificada en la clase: {ganadora}")
    print("=====" * 10, "\n")

print("Entrenamiento completado")
print(f"Número final de clases: {pesos_f.shape[0]}")

import numpy as np

# Aplanar las características
X_train_flat = resultados.reshape(resultados.shape[0], -1)
print(f"X_train_flat shape: {X_train_flat.shape}")

# Parámetros de la red
n = X_train_flat.shape[1]  # Número de características (10 * 28 = 280)
m = 3  # Número inicial de neuronas en la capa de salida (una por cada clase)

# Inicialización de pesos
pesos_f = np.ones((m, n)) * (1 / (1 + n))
pesos_b = np.ones((n, m))

# Parámetros de vigilancia
p = 0.1
limite = 10

print(f"Pesos f shape: {pesos_f.shape}")
print(f"Pesos b shape: {pesos_b.shape}")

# Para almacenar las predicciones
predicted_labels = []

# Fase de entrenamiento y clasificación
for i in range(X_train_flat.shape[0]):
    sample = X_train_flat[i]

    F1 = sample

    k = []
    for j in range(pesos_f.shape[0]):
        u_sum_f = np.dot(F1, pesos_f[j])
        k.append((u_sum_f, j))

    k_g = [index for _, index in sorted(k, reverse=True)]

    flag = False
    for idx in k_g:
        u_sum_b = np.dot(F1, pesos_b[:, idx])
        sum_F1 = np.sum(F1)
        R = u_sum_b / sum_F1
        if R >= p:
            flag = True
            ganadora = idx
            break

    if not flag:
        ganadora = pesos_f.shape[0]
        if ganadora < limite:
            Ncol_f = F1.copy()
            Ncol_b = np.ones(F1.size)

            pesos_f = np.vstack([pesos_f, Ncol_f])
            pesos_b = np.column_stack([pesos_b, Ncol_b])

            print(f"Nueva neurona añadida. Total clases: {pesos_f.shape[0]}")

    pesos_b[:, ganadora] = pesos_b[:, ganadora] * F1
    pesos_f[ganadora] = pesos_b[:, ganadora] / (0.5 + np.sum(pesos_b[:, ganadora]))

    predicted_labels.append(ganadora)
    print(f"Muestra {i} clasificada en la clase: {ganadora}.")

print("\nEntrenamiento y clasificación completados")
print(f"Número final de clases creadas por la red: {pesos_f.shape[0]}")

# Aplanar los resultados de la CNN
X_aplanado = np.array([imagen.flatten() for imagen in resultados])
X = X_aplanado

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
delta = 0.01
X0 = 1
ECM = 10
epocas = 2000
epoca = 0
alpha = 0.2

# Asumiendo que X es tu conjunto de datos de entrada
# X = ...

# Etiquetas
Yd = np.array([[1,0,0]] * 25 + [[0,1,0]] * 25 + [[0,0,1]] * 25)

np.random.seed(40)
neuronas_entrada = 120
neuronas_ocultas = 60
neuronas_salida = 3  # 3 para las tres clases de frutas

# Capa de entrada
W_entrada = 2*np.random.rand(X.shape[1], neuronas_entrada) - 1

# Capa oculta
W_oculta = 2*np.random.rand(neuronas_entrada+1, neuronas_ocultas) - 1

# Capa salida
W_salida = 2*np.random.rand(neuronas_ocultas+1, neuronas_salida) - 1

# Inicialización de pesos
W_entrada_new = np.zeros(W_entrada.shape)
W_entrada_old = np.zeros(W_entrada.shape)

W_oculta_new = np.zeros(W_oculta.shape)
W_oculta_old = np.zeros(W_oculta.shape)

W_salida_new = np.zeros(W_salida.shape)
W_salida_old = np.zeros(W_salida.shape)

# Funciones de activación y sus derivadas
def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def deriv_sigmoide(z):
    return np.exp(-z)/(1.0+np.exp(-z))**2

def tang_hiper(z):
    return np.tanh(z)

def deriv_tang_hiper(z):
    return 1 - np.tanh(z)**2

ECM_camb = []
while (ECM >= 0.001 and epoca <= epocas):
    # Propagación hacia adelante
    z_entrada = np.dot(X, W_entrada)
    y_entrada = np.insert(tang_hiper(z_entrada), 0, -1, axis=1)

    z_oculta = np.dot(y_entrada, W_oculta)
    y_oculta = np.insert(sigmoide(z_oculta), 0, -1, axis=1)

    z_salida = np.dot(y_oculta, W_salida)
    Yobt = sigmoide(z_salida)

    # Retropropagación
    aux3 = (Yd - Yobt) * deriv_sigmoide(z_salida)
    aux2 = np.dot(aux3, W_salida[1:,:].T) * deriv_sigmoide(z_oculta)
    aux = np.dot(aux2, W_oculta[1:,:].T) * deriv_tang_hiper(z_entrada)

    # Actualización de pesos con Momentum
    W_salida_new = W_salida + delta * np.dot(y_oculta.T, aux3) + alpha * (W_salida - W_salida_old)
    W_salida_old = W_salida
    W_salida = W_salida_new

    W_oculta_new = W_oculta + delta * np.dot(y_entrada.T, aux2) + alpha * (W_oculta - W_oculta_old)
    W_oculta_old = W_oculta
    W_oculta = W_oculta_new

    W_entrada_new = W_entrada + delta * np.dot(X.T, aux) + alpha * (W_entrada - W_entrada_old)
    W_entrada_old = W_entrada
    W_entrada = W_entrada_new

    # Cálculo del Error Cuadrático Medio
    ECM = np.mean((Yd - Yobt)**2)
    ECM_camb.append(ECM)
    epoca += 1

    print(f"Epoca: {epoca}")
    print(f"Y obt de la epoca (primeras 5 muestras):")
    print(np.round(Yobt[:5], 3))
    print(f"Error cuadratico medio: {ECM:.6f}\n")

print("Entrenamiento completado.")
print("Y obt final (primeras 5 muestras):")
print(np.round(Yobt[:5], 3))

# Convertir las probabilidades a clases
clases_predichas = np.argmax(Yobt, axis=1)
clases_reales = np.argmax(Yd, axis=1)

print("\nClasificación final (todas las 75 muestras):")
for i in range(75):
    fruta_predicha = ["Manzana", "Plátano", "Tuna"][clases_predichas[i]]
    fruta_real = ["Manzana", "Plátano", "Tuna"][clases_reales[i]]
    print(f"Muestra {i+1:2d}: Predicción: {fruta_predicha:7s}, Real: {fruta_real:7s} | Probabilidades: {Yobt[i]}")

# Calcular precisión
precision = np.mean(clases_predichas == clases_reales)
print(f"\nPrecisión del modelo: {precision:.2%}")

# Calcular precisión por clase
for i, fruta in enumerate(["Manzana", "Plátano", "Tuna"]):
    precision_clase = np.mean((clases_predichas == i) & (clases_reales == i))
    total_clase = np.sum(clases_reales == i)
    correctas_clase = np.sum((clases_predichas == i) & (clases_reales == i))
    print(f"Precisión de {fruta:7s}: {precision_clase:.2%} ({correctas_clase}/{total_clase})")

plt.figure(figsize=(10, 6))
plt.plot(range(1, epoca + 1), ECM_camb)
plt.xlabel('Época')
plt.ylabel('Error Cuadrático Medio (ECM)')
plt.title('ECM por época')
plt.grid(True)
plt.show()

print(np.round(Yobt))

# Predicción con nuevos datos
def predecir(X_nuevos, W_entrada, W_oculta, W_salida):
    # Propagación hacia adelante con los nuevos datos
    z_entrada = np.dot(X_nuevos, W_entrada)
    y_entrada = np.insert(tang_hiper(z_entrada), 0, -1, axis=1)

    z_oculta = np.dot(y_entrada, W_oculta)
    y_oculta = np.insert(sigmoide(z_oculta), 0, -1, axis=1)

    z_salida = np.dot(y_oculta, W_salida)
    Yobt = sigmoide(z_salida)

    # Convertir las probabilidades a clases
    clases_predichas = np.argmax(Yobt, axis=1)

    return clases_predichas, Yobt

# Ejemplo de predicción con nuevos datos
X_nuevos = np.array([...])  # Aquí irían los nuevos datos de entrada
clases_predichas, probabilidades = predecir(X_nuevos, W_entrada, W_oculta, W_salida)

# Mostrar resultados de la predicción
for i, clase in enumerate(clases_predichas):
    fruta_predicha = ["Manzana", "Plátano", "Tuna"][clase]
    print(f"Predicción para muestra {i+1}: {fruta_predicha}, Probabilidades: {probabilidades[i]}")


