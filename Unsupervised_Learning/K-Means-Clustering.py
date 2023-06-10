import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data
"""
# 0  #1   #2   #3  => cada uno de estas columnas es una caracteristica que podemos ver en iris.DESCR. cm de la flor etc.
[[5.1 3.5 1.4 0.2] #  iris 1
 [4.9 3.  1.4 0.2] #  iris 2
 [4.7 3.2 1.3 0.2] #   ....
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 ....
 Cada una de estas iris pertenece a una especie de 3 posibles
"""

"""
Aqui seleccionamos data que nos interesa (columnas 0 y columna 1) aunque podriamos haber cogido otras columnas (u otros features)
"""
x = samples[:,0]
y = samples[:,1]

sepal_length_width = np.array(list(zip(x, y))) # Juntamos las features  

# Step 1: Place K random centroids. 
k = 3 # Number of centroidsd

# numpy nos ayuda a seleccionar un valor entre el minimo y el maximo de las columnas para generar un centroide aleatorio
centroids_x = np.random.uniform(min(x), max(x), size=k) # numpy nos ayuda a seleccionar un valor entre el minimo y el maximo de las columnas para generar un centroide aleatorio
centroids_y = np.random.uniform(min(y), max(y), size=k)
centroids = np.array(list(zip(centroids_x, centroids_y))) # juntamos los centroides en un array  
print(centroids)
"""
[[6.83703584 3.2858285 ]
 [7.76659513 3.34353882]
 [6.79484898 3.19893215]]
 """

# Step 2: Assign samples to nearest centroid
# Distance formula so we know which centroid is the closest one to our sample
def distance(a, b):
  x = (a[0] - b[0]) **2 # (X1 - X2)**2
  y = (a[1] - b[1]) **2 # (Y1 - Y2)**2
  distance = (x+y) ** 0.5  
  return distance
"""
Esta funcion la vamos a usar para un sample (punto) y cada centroid. Cada uno de ellos tiene 2 valores que corresponden a los features de las columnas seleccionadas al principio 
([6.83703584 3.2858285 ] para el primer centroid por ejemplo) y que actuan como sus coordenadas x e y. 
Por lo tanto, tenemos que averiguar la distancia para estos dos por separado (Euclidian method)
"""

# Cluster labels for each point (either 0, 1, or 2)
labels = np.zeros(len(samples))
print(labels)
"""
Esta es la lista donde luego pondremos a que centroid esta mas cerca cada sample que pasemos 
Por lo tanto, habra tantos labels (o centroid mas cercano al sample) como samples haya = len(sample)
"""

# A function that assigns the nearest centroid to a sample
def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k) # 
  """
  aqui igual, crearemos una lista con las distancias a los 3 centroides para cada sample. Asi que necesitamos una lista con tantos ceros como centroides
  Para cada sample vamos a crear una lista de 3 elementos distance = [distancia al centroid 1, distancia al centroid 2, distancia al centroid 3]
  """
  for i in range(k):
    distances[i] = distance(sample, centroids[i])
  closest_centroid = np.argmin(distances) 
  """
  np.argmin da el index del valor minimo de la lista. De este modo si las distancias al centroid 1 2 y 3 para el sample fueran dist= [2, 4, 1] retornaria 3 (index de la distancia menor)
  o lo que es lo mismo el centroid que esta mas cerca al sample
  """
  return closest_centroid

# Assign the nearest centroid to each sample
"""
hemos hecho distancia a 3 centroids para 1 sample. Ahora tenemos que hacerlo esto con todos los samples que tengamos 
"""
for i in range(len(samples)):
  labels[i] = assign_to_centroid(samples[i], centroids) # asi si para el sample 1 fuera el centroid mas cercano el 2, para el sample 2 fuera 1 tendriamos lables = [2, 1, ...]

# Print labels
print(labels)

