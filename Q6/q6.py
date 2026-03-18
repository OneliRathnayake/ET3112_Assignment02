import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
c_tls = mean[1] - m_tls * mean[0]

# Line
x_line = np.linspace(min(x), max(x), 100)
y_line = m_tls * x_line + c_tls

# Equation
eq = f"y = {m_tls:.3f}x + {c_tls:.3f}"

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label="Data Points")
plt.plot(x_line, y_line, color='green', label=eq)

plt.legend()
plt.title("Total Least Squares Fit")
plt.gca().invert_yaxis()
plt.show()