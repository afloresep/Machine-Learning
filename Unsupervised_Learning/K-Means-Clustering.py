import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

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
k = 3 # Number of centroids

 #numpy nos ayuda a seleccionar un valor entre el minimo y el maximo de las columnas para generar un centroide aleatorio
centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)
centroids = np.array(list(zip(centroids_x, centroids_y)))


"""
ejemplo de centroid generado
[[6.83703584 3.2858285 ]
 [7.76659513 3.34353882]
 [6.79484898 3.19893215]]
 """

 # Distance formula so we know which centroid is the closest one to our sample
def distance(a, b):
  one = (a[0] - b[0]) ** 2
  two = (a[1] - b[1]) ** 2
  distance = (one + two) ** 0.5
  return distance
"""
Esta funcion la vamos a usar para un sample (punto) y cada centroid. Cada uno de ellos tiene 2 valores que corresponden a los features de las columnas seleccionadas al principio 
([6.83703584 3.2858285 ] para el primer centroid por ejemplo) y que actuan como sus coordenadas x e y. 
Por lo tanto, tenemos que averiguar la distancia para estos dos por separado (Euclidian method)
"""

# A function that assigns the nearest centroid to a sample
def assign_to_centroid(sample, centroids):
  k = len(centroids)
  distances = np.zeros(k)
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


# To store the value of centroids when it updates
centroids_old = np.zeros(centroids.shape)

# Cluster labeles (either 0, 1, or 2)
labels = np.zeros(len(samples))
"""
Esta es la lista donde luego pondremos a que centroid esta mas cerca cada sample que pasemos 
Por lo tanto, habra tantos labels (o centroid mas cercano al sample) como samples haya = len(sample)
"""

distances = np.zeros(k)

# Initialize error:
error = np.zeros(k)

for i in range(k):
  error[i] = distance(centroids[i], centroids_old[i])

# Repeat Steps 2 and 3 until convergence:
while error.all() != 0:
  # Step 2: Assign samples to nearest centroid
  for i in range(len(samples)):
    labels[i] = assign_to_centroid(samples[i], centroids)
  """
  aqui vamos a por un label (centroid mas cercano) para cada sample (iterado por i) y gracias a la funcion assign_to_centroid
  hemos hecho distancia a 3 centroids para 1 sample. Ahora tenemos que hacerlo esto con todos los samples que tengamos 
  asi se quedara una lista donde en vez de las coordenadas (x,y) tenemos valores 0,1 o 2 si se acerca más al cluster 1, 2 o 3 respectivamente 
  """
  # Step 3: Update centroids
  centroids_old = deepcopy(centroids) # guardamos el anterior valor porque queremos compararlo
  for i in range(k):
     points = [] # lista vacia donde vamos a meter los puntos (samples) que comparten el k más cercano
     for j in range(len(sepal_length_width)): 
        if labels[j] == i:
          points.append(sepal_length_width[j])
     centroids[i] = np.mean(points, axis=0)
     error[i] = distance(centroids[i], centroids_old[i])
  """
    labels es [0, 0, 1, 2, 0, 0, ...] donde cada valor es el centroid mas cercano de cada sample (para el sample 1 es el centroid 0, para el 2 el cero tambien, para el 3 el centroid 1 etc...)
    aqui basicamente estamos metiendo en una lista todos samples (sepal_length_width[j] que tengan un label (centroid mas cercano) igual a i (primero con el centroid 0, luego con el 1 y luego con el 2))
    de este modo vamos a agrupar todos los samples en una lista llamada points (lista de arrays porque recordamos que los samples tienen 2 valores)
    despues de esta lisat points vamos a sacar el mean (la media) y lo metemos en la lista centroids (media de todos los samples que tienen mas cerca al centroid 0, luego media de los que tienen cerca al centroid 1 etc)
    hasta completar la lista centroids. Asi tenemos dos centroids
    el old centroid = punto aleatorio generado por un valor aleatorio entre el min y max gracias a la funcion de np. y el nuevo centroid creado a partir de la media de los valores de todos los datos 
    que estaban cerca del old centroid
  """

# Visualizar
colors = ['r', 'g', 'b']

for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

plt.show()
