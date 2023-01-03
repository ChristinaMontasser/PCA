import numpy as np
import cv2
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
      self.n_components = n_components

    def fit(self, X):
      self.X = X
      X_meaned = self.X - np.mean(self.X , axis = 0)
      cov_mat = np.cov(X_meaned , rowvar = False)   
      self.eigen_values , self.eigen_vectors = np.linalg.eigh(cov_mat)
      sorted_index = np.argsort(self.eigen_values)[::-1]
      self.sorted_eigenvalue = self.eigen_values[sorted_index]
      #similarly sort the eigenvectors 
      self.sorted_eigenvectors = self.eigen_vectors[:,sorted_index]
      self.eigenvector_subset = self.sorted_eigenvectors[:,0:self.n_components]
      self.X_reduced = np.dot(self.eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
      return self.X_reduced

    def transform_inverse(self, X_reduced):
      new = np.dot(self.eigenvector_subset, X_reduced.transpose()).transpose() +np.mean(self.X, axis=0)
      return new.reshape(self.X.shape[0], int(self.X.shape[1]/3), 3)

def draw_image(X):
  plt.figure(figsize=(15, 40))
  plt.imshow(X)
  plt.imshow(X.astype("uint8"))

