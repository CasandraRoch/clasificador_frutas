# Clasificador de Frutas con Deep Learning desde Cero

Proyecto de clasificación de imágenes de frutas (manzana, plátano 
y tuna) implementando una red neuronal profunda completamente 
manual con NumPy, sin usar frameworks como TensorFlow o PyTorch.

## ¿Qué hace este proyecto?

1. Carga imágenes de 3 categorías de frutas desde Google Drive
2. Preprocesa cada imagen: escala de grises, redimensión a 64×64 
   y normalización
3. Aplica 2 capas convolucionales manuales con 4 kernels de 
   detección de bordes
4. Clasifica las imágenes usando dos enfoques distintos:
   - Red neuronal con backpropagation (supervisado)
   - Red ART — Adaptive Resonance Theory (no supervisado)
5. Reporta la precisión por clase: manzana, plátano y tuna

## Arquitectura del pipeline
Se implementaron 4 kernels manuales 3×3:
- Bordes verticales (Kernel 1)
- Bordes horizontales (Kernel 2)  
- Diagonal principal (Kernel diagonal 1)
- Diagonal inversa (Kernel diagonal 2)

## Tecnologías

- Python 3 / Google Colab
- NumPy — convolución, backpropagation y ART desde cero
- Pillow — carga y preprocesamiento de imágenes
- Matplotlib — visualización de imágenes y curva de ECM
- Google Drive — almacenamiento del dataset

## Dataset

75 imágenes en total: 25 por cada clase (manzana, plátano, tuna),
organizadas en carpetas separadas en Google Drive.

## Aprendizajes

- Implementación manual de convolución 2D y max pooling
- Función de activación ReLU y su papel en la extracción de bordes
- Backpropagation con dos funciones de activación distintas 
  (tanh en capa oculta, sigmoide en salida)
- Algoritmo ART para clasificación no supervisada
- Pipeline completo de visión artificial sin frameworks
