import cv2 as cv
import numpy as np

# Load image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# TLS
X = np.vstack((x, y)).T
mean = np.mean(X, axis=0)
X_centered = X - mean

U, S, Vt = np.linalg.svd(X_centered)

direction = Vt[0]
dx, dy = direction

m_tls = dy / dx

# Angle
theta_tls = np.arctan(m_tls) * 180 / np.pi
print("Estimated Angle (TLS):", theta_tls, "degrees")