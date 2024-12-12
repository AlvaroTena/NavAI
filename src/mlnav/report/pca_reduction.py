from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def pca_reduction(model, data):
    pca = PCA(n_components=2)
    X = pca.fit_transform(data)
    y = model.fit_predict(X)
    
    #Getting the Centroids
    centroids = model.cluster_centers_
    u_labels = np.unique(y)
    
    #plotting the results:    
    for i in u_labels:
        plt.scatter(X[y == i , 0] , X[y == i , 1] , label = i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'black')
        plt.legend()
        plt.show()
    
    return plt