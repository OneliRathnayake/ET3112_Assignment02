import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Give your image path here
image_path = r"D:\Assignment 2\ET3112_Assignment02\1C.jpg"  # change this

# Read image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded properly
if img is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Edge detection
edges = cv2.Canny(img, 100, 200)

# Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

if lines is not None:
    rho, theta = lines[0][0]
    theta_hough = np.degrees(theta) - 90
    print("Hough Transform Angle (degrees):", theta_hough)

    # Convert polar to Cartesian
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Plot result
    plt.figure(figsize=(8, 5))
    plt.imshow(img, cmap='gray')
    plt.plot([x1, x2], [y1, y2], 'r', linewidth=2)
    plt.title(f"Hough Line (θ ≈ {theta_hough:.2f}°)")
    plt.axis("off")
    plt.show()

else:
    print("No lines detected. Try adjusting parameters.")

    plt.figure(figsize=(8, 5))
    plt.imshow(edges, cmap='gray')
    plt.title("Edges (No lines detected)")
    plt.axis("off")
    plt.show()