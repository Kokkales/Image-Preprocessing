import cv2
import numpy as np
import matplotlib.pyplot as plt

# import image
image = cv2.imread('./images-project-1/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# image dimensions
height, width = image.shape[:2]

# noise density "salt & pepper"
noise_density = 0.05

# create noise
salt_and_pepper = np.random.rand(height, width)

# apply noise "salt & pepper"
image_with_noise = image.copy()
image_with_noise[salt_and_pepper < noise_density / 2] = 0
image_with_noise[salt_and_pepper > 1 - noise_density / 2] = 255


# Εμφανίζουμε τις εικόνες στον ίδιο γράφο
plt.figure(figsize=(10, 5))

# plot initial image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# plot image with noise"salt & pepper"
plt.subplot(1, 2, 2)
plt.imshow(image_with_noise, cmap='gray')
plt.title('Image with Salt & Pepper Noise')
plt.axis('off')

plt.show()

# C https://medium.com/@turgay2317/image-processing-in-python-with-opencv-blur-3e474fda6a52
# apply mean filter 3x3
filtered_image_3x3 = cv2.blur(image_with_noise, (3, 3))

# apply mean filter 5x5
filtered_image_5x5 = cv2.blur(image_with_noise, (5, 5))

# apply mean filter 7x7
filtered_image_7x7 = cv2.blur(image_with_noise, (7, 7))

# Plot all images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_with_noise, cmap='gray')
plt.title('Image with Salt & Pepper Noise')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(filtered_image_3x3, cmap='gray')
plt.title('Filtered Image (3x3)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_image_5x5, cmap='gray')
plt.title('Filtered Image (5x5)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filtered_image_7x7, cmap='gray')
plt.title('Filtered Image (7x7)')
plt.axis('off')

plt.show()