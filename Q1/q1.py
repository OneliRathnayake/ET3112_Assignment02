import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the cropped image (Fig 1c)
img = cv.imread("1C.jpg", cv.IMREAD_GRAYSCALE)

# Apply Canny edge detector
edges = cv.Canny(img, 550, 690)

# Extract edge coordinates (given in the assignment)
indices = np.where(edges != [0])

x = indices[1]
y = indices[0]

# Display the images
plt.figure(figsize=(10,5))

# Original image
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Edge image
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

plt.show()