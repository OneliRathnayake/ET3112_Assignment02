import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv.Canny(img, 550, 690)

# Extract edge coordinates (given code in assignment)
indices = np.where(edges != [0])

x = indices[1]
y = indices[0]

# Combine x and y into a single array
points = np.column_stack((x, y))

# Compute centroid of the points
centroid = np.mean(points, axis=0)

# Center the data
centered_points = points - centroid

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(centered_points)

# Direction vector of TLS line
direction = Vt[0]

dx = direction[0]
dy = direction[1]

# Create line for plotting
t = np.linspace(-1000, 1000, 1000)

line_x = centroid[0] + t * dx
line_y = centroid[1] + t * dy

# Plot scatter and TLS line
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(line_x, line_y, color="red", linewidth=2, label="Total Least Squares Fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Total Least Squares Fit Line")
plt.legend()

# Match image coordinate system
plt.gca().invert_yaxis()

plt.show()