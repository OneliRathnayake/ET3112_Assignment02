import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


# Load cropped crop image
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", 0)   # Fig 1c (grayscale)


# Apply Canny edge detection
edges = cv.Canny(img, 550, 690)


# Extract edge coordinates
indices = np.where(edges != 0)

x = indices[1]
y = indices[0]

# reshape for sklearn
X = x.reshape(-1, 1)
Y = y


# 4. Apply RANSAC
ransac = RANSACRegressor(LinearRegression())

ransac.fit(X, Y)

# predicted line
y_ransac = ransac.predict(X)

# Plot scatter + RANSAC line
plt.figure(figsize=(6,6))

plt.scatter(x, y, s=1, label="Edge Points")
plt.plot(x, y_ransac, color='red', linewidth=2, label="RANSAC Fit")

plt.title("RANSAC Line Estimation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.gca().invert_yaxis()


#Save image for report
plt.savefig("q10_ransac_line.png", dpi=300)

plt.show()


# Calculate angle (for Q11)
slope = ransac.estimator_.coef_[0]

theta = np.degrees(np.arctan(slope))

print("Estimated slope:", slope)
print("Estimated crop field angle (degrees):", theta)