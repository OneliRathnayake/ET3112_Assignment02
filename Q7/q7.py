import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv.Canny(img, 550, 690)

# Extract edge coordinates
indices = np.where(edges != [0])

x = indices[1]
y = indices[0]

# Combine points
points = np.column_stack((x, y))

# Compute centroid
centroid = np.mean(points, axis=0)

# Center the data
centered_points = points - centroid

# Perform SVD
U, S, Vt = np.linalg.svd(centered_points)

# Direction vector of TLS line
direction = Vt[0]

dx = direction[0]
dy = direction[1]

# Calculate slope
slope = dy / dx

# Calculate crop field angle
theta_rad = np.arctan(slope)
theta_deg = np.degrees(theta_rad)

print("TLS Slope:", slope)
print("Estimated Crop Field Angle (degrees):", theta_deg)