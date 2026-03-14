import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1(c).png", cv.IMREAD_GRAYSCALE)

# Display and save original image
plt.figure()
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.savefig("original_image.png", dpi=300)
plt.show()

# Apply Canny Edge Detector
edges = cv.Canny(img, 550, 690)

# Display and save edge image
plt.figure()
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.savefig("canny_edges.png", dpi=300)
plt.show()

# Extract edge coordinates
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

# Scatter plot
plt.figure()
plt.scatter(x, y, s=1)
plt.title("Scatter Plot of Edge Points")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().invert_yaxis()

plt.savefig("scatter_plot.png", dpi=300)
plt.show()

# least squares fit
m, b = np.polyfit(x, y, 1)

# line
y_fit = m*x + b

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1)
plt.plot(x, y_fit, color='red', linewidth=2)

plt.title("Least Squares Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().invert_yaxis()

plt.savefig("q3_least_squares.png", dpi=300)
plt.show()

#q4 Estimated angle (Least Squares)
theta = np.degrees(np.arctan(m))
print("Estimated angle (Least Squares):", theta)

#q6 Total Least Squares
#stack coordinates
data = np.vstack((x, y))

# mean
mean = np.mean(data, axis=1)

# centered data
data_centered = data - mean[:, np.newaxis]

# SVD
U, S, Vt = np.linalg.svd(data_centered)

# direction vector
direction = U[:,0]

m_tls = direction[1] / direction[0]

b_tls = mean[1] - m_tls * mean[0]

# line
y_tls = m_tls*x + b_tls

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1)
plt.plot(x, y_tls, color='green', linewidth=2)

plt.title("Total Least Squares Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().invert_yaxis()

plt.savefig("q6_tls.png", dpi=300)
plt.show()