import cv2 as cv
import numpy as np
from sklearn.linear_model import RANSACRegressor

# Load image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# RANSAC
x_reshaped = x.reshape(-1, 1)

ransac = RANSACRegressor()
ransac.fit(x_reshaped, y)

# Slope
m_ransac = ransac.estimator_.coef_[0]

# Angle
theta_ransac = np.arctan(m_ransac) * 180 / np.pi
print("Estimated Angle (RANSAC):", theta_ransac, "degrees")