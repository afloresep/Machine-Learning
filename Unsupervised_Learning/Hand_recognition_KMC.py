import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
 
digits = datasets.load_digits()
# print(digits.DESCR, digits.data, digits.target)
samples = digits.data

# plt.gray()
# plt.matshow(digits.images[100])
# plt.show()


# Figure size (width, height)
 
# fig = plt.figure(figsize=(6, 6))
 
# Adjust the subplots 
 
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# For each of the 64 images
 
# for i in range(64): 
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i1-th position
    #  ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th positiona
    #  ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
 
    # Label the image with the target value 
    #  ax.text(0, 7, str(digits.target[i]))
# plt.show()


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 10, random_state = 100, n_init = 10) # 10 different numbers (0-9) so clusters (or number of groups) should be k = 10
model.fit(samples)

print(model.cluster_centers_[1].reshape(8,8)) 



fig = plt.figure(figsize=(10,5)) # Because data samples live in 64-dimensional space (8x8), the centroids have values so they can be images. First we add a figure of size 8x3 using .figure()
# Esta va a ser la figura donde van a estar los centroids reflejados. Cada uno de ellos 8x8
fig.suptitle('Cluster Center Images', fontsize= 14, fontweight = 'bold') # Cluster Center = centroids

for i in range(10): 
  ax = fig.add_subplot(2,5,1+i) # dentro de fig vamos a crear 10 imagenes (cada centroid) y las vamos a mostrar todas juntas en una disposición 2x5 (5 imagenes en 1 filas y 5 imagenes en otra fila )
  
  # Initialize subplots in a grid of 2x5, at i+1th position
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap= plt.cm.binary) 
"""
plt.cm.binary is a color map provided by matplotlib. It represents a binary color map where the values are mapped to a black and white gradient. 
Values closer to zero are displayed as black, while values closer to one are displayed as white. This color map is often used when visualizing binary or grayscale images.
KMeans almacena los centroids en model.cluster_centers_. Son 10 arrays con 64 valores cada uno. Ahora tiene un shape de 4x16 y queremos que sea 8x8 asi que usamos al funcion reshape(8,8) 

"""
plt.show()



