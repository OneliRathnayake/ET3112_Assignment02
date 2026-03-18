import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# RANSAC
x_reshaped = x.reshape(-1, 1)

ransac = RANSACRegressor()
ransac.fit(x_reshaped, y)

y_ransac = ransac.predict(x_reshaped)

# Get slope & intercept
m_ransac = ransac.estimator_.coef_[0]
c_ransac = ransac.estimator_.intercept_

# Equation
eq = f"y = {m_ransac:.3f}x + {c_ransac:.3f}"

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label="Data Points")
plt.plot(x, y_ransac, color='orange', label=eq)

plt.legend()
plt.title("RANSAC Fit")
plt.gca().invert_yaxis()
plt.show()