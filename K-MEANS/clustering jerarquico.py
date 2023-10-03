#!/usr/bin/env python
# coding: utf-8

# In[71]:


pip install numpy scikit-learn opencv-python


# In[265]:


import numpy as np
from PIL import Image
import random
import cv2
from scipy.cluster.hierarchy import linkage, dendrogram


# In[266]:


# Se utiliza distancia euclidiana para calcular distancia entre los puntos
def dist_euclidiana(a, b):
    a=a.astype(np.float64)
    b=b.astype(np.float64)
    return np.sqrt(np.sum((a - b) ** 2))


# In[267]:


i = cv2.imread('C:/Users/Usuario/Downloads/imagen1.jpg')
i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)


# In[268]:


import matplotlib.pyplot as plt


# In[269]:


plt.figure(figsize=(10,8))
plt.imshow(i)
plt.title('imagen original')
plt.show()


# In[270]:


# submuestreo de la imagen para reducir la calidad y cantidad de pixeles
fs = 0.1
i_submuestra = cv2.resize(i, None, fx=fs, fy=fs)


# In[271]:


plt.figure(figsize=(10,8))
plt.imshow(i_submuestra)
plt.title('submuestra de la imagen original')
plt.show()


# In[272]:


#Se da forma a la matriz para ser utilizada
alto, ancho, canales = i_submuestra.shape
alto, ancho, canales = i.shape

# Redefine la matriz de píxeles
datos = i_submuestra.reshape(-1, canales)
datos_original=i.reshape(-1, canales)
num_puntos = datos.shape[0]
num_puntos_original = datos_original.shape[0]


# In[273]:


num_puntos


# In[274]:


num_puntos_original #es posible comprobar que efectivamente se redujo de forma considerable el numero de puntos


# In[275]:


n_pixeles = min(len(datos), 20) #elije el minimo entre el largo de la matriz y 40, como son mas de 40, elije 20 al azar.
indices = np.random.choice(len(datos), n_pixeles, replace=False)
datos_muestra = datos[indices]


# In[276]:


n_pixeles


# In[277]:


indices


# In[278]:


datos_muestra


# In[279]:


# Calcula la matriz de distancias entre los puntos
def calcular_matriz_distancias(datos_muestra):
    n = n_pixeles
    distancias = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distancias[i, j] = np.linalg.norm(datos[i] - datos[j])
    return distancias


# In[297]:


# Implementación del clustering jerárquico
def clustering_jerarquico(datos_muestra):
    n = n_pixeles
    distancias = calcular_matriz_distancias(datos_muestra)
    clusters = [[i] for i in range(n)]

    while len(clusters) > 6: #6 colores
        matriz_enlace = np.zeros((len(clusters), len(clusters)))

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distancia_promedio = np.mean([distancias[p1, p2] for p1 in clusters[i] for p2 in clusters[j]])
                matriz_enlace[i, j] = matriz_enlace[j, i] = distancia_promedio

        min_distancia = np.min(matriz_enlace)
        i, j = np.where(matriz_enlace == min_distancia)

        i, j = i[0], j[0]

        nuevo_cluster = clusters[i] + clusters[j]
        del clusters[j]
        clusters[i] = nuevo_cluster

    return clusters


# In[298]:


# Ejecuta el clustering jerárquico
resultados_clustering = clustering_jerarquico(datos_muestra)
resultados_clustering


# In[299]:


#enlace para generar dendograma
enlace = linkage(datos_muestra, method='ward')


# In[300]:


# Genera el dendograma
dendrogram(enlace, orientation='top', labels=range(1, len(datos_muestra) + 1))
plt.xlabel('Índice de datos')
plt.ylabel('Distancia')
plt.title('Dendograma de Clustering Jerárquico')
plt.show()


# In[301]:


# Calcular los colores para los 3 clústeres finales
colores = [np.mean(datos_muestra[cluster], axis=0).astype(int) for cluster in resultados_clustering]


# In[302]:


colores


# In[310]:


def asigna_colores(p):
    distances = [np.linalg.norm(p - color) for color in colores]
    closest_cluster_index = np.argmin(distances)
    return colores[closest_cluster_index]


# In[307]:


asigna_colores(datos_muestra)


# In[308]:


#Se da forma a la matriz para ser utilizada
alto, ancho = datos_muestra.shape
imagen_reconstruida = np.apply_along_axis(asigna_colores, 1, datos_muestra).reshape(20, 3, 1)


# In[309]:


# Mostrar la imagen reconstruida
plt.figure(figsize=(8,7))
plt.imshow(reconstruccion)
plt.title('Imagen Reconstruida')
plt.show()


# In[ ]:




