import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load image from local file ---
filename ="C:\\Users\\user\\Pictures\\Screenshots\\Screenshot 2026-03-13 214524.png"  # Replace with your image path
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read the image. Check the file path.")
    exit()

# --- Edge detection ---
edges = cv2.Canny(img, 100, 200)

# --- Hough Transform ---
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

# --- Plotting ---
plt.figure(figsize=(8,5))
plt.imshow(img, cmap='gray')
plt.axis("off")

if lines is not None:
    # Take the first detected line for demonstration
    rho, theta = lines[0][0]
    theta_hough = np.degrees(theta) - 90
    print("Hough Transform Angle (degrees):", theta_hough)

    # Convert polar coordinates to Cartesian
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    # Draw the line
    plt.plot([x1, x2], [y1, y2], 'r', linewidth=2)
    plt.title(f"Hough Line (θ ≈ {theta_hough:.2f}°)")
else:
    print("No lines detected. Adjust Canny/Hough parameters.")
    plt.title("Edges (No lines detected)")
    plt.imshow(edges, cmap='gray')

plt.show()