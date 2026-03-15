import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the cropped image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv.Canny(img, 550, 690)

# Extract edge coordinates
indices = np.where(edges != [0])

x = indices[1]
y = indices[0]

# Reshape x for regression
x_reshaped = x.reshape(-1,1)

# Least Squares Regression
model = LinearRegression()
model.fit(x_reshaped, y)

# Get slope and intercept
m = model.coef_[0]
c = model.intercept_

# Calculate crop field angle
theta_rad = np.arctan(m)
theta_deg = np.degrees(theta_rad)

print("Slope (m):", m)
print("Estimated Crop Field Angle (degrees):", theta_deg)

# Plot scatter and fitted line
y_pred = model.predict(x_reshaped)

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label="Edge Points")

sorted_indices = np.argsort(x)
plt.plot(x[sorted_indices], y_pred[sorted_indices],
         color="red", linewidth=2, label="Least Squares Fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Least Squares Fit with Crop Field Angle")
plt.legend()

plt.gca().invert_yaxis()

plt.show()