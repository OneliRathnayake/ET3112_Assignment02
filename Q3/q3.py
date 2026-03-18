import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 550, 690)

indices = np.where(edges != [0])
x = indices[1]
y = indices[0]

# Fit
m, c = np.polyfit(x, y, 1)
y_fit = m * x + c

# Equation text
eq = f"y = {m:.3f}x + {c:.3f}"

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=1, label="Data Points")
plt.plot(x, y_fit, color='red', label=eq)

plt.legend()
plt.title("Least Squares Fit")
plt.gca().invert_yaxis()
plt.show()

