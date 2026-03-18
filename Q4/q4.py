import cv2 as cv
import numpy as np

# Load image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# Least Squares
m, c = np.polyfit(x, y, 1)

# Angle
theta = np.arctan(m) * 180 / np.pi
print("Estimated Angle (Least Squares):", theta, "degrees")