import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read image (CHANGE PATH)
img = cv.imread(r"D:\Assignment 2\ET3112_Assignment02\1C.jpg", cv.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found")
    exit()

# Canny Edge Detection (MUST USE THESE VALUES)
edges = cv.Canny(img, 550, 690)

# Display images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Edge Image")
plt.imshow(edges, cmap='gray')

plt.show()