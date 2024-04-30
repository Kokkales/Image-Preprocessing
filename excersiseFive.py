import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv2.imread("./images-project-1/lenna.bmp")

# Define the mean and standard deviation of the Gaussian noise
mean = 0
std_dev = 0.5

# Generate Gaussian noise with the specified mean and standard deviation
noise = np.random.normal(mean, std_dev, img.shape).astype(img.dtype)  # Ensure same data type

# Add the noise to the image
noisy_img = cv2.add(img, noise)

# Clip the pixel values to be between 0 and 255 (assuming uint8 image)
noisy_img = np.clip(noisy_img, 0, 255)
# Εμφάνιση των εικόνων πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.show()
