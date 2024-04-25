import cv2
import numpy as np
import matplotlib.pyplot as plt

# Διαβάζουμε την εικόνα σε grayscale
image = cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Διαστάσεις της εικόνας
height, width = image.shape[:2]

# Ποσοστό κάλυψης για τον θόρυβο "salt & pepper"
noise_density = 0.05

# Δημιουργούμε μια τυχαία διάταξη θορύβου
salt_and_pepper = np.random.rand(height, width)

# Εφαρμόζουμε τον θόρυβο "salt & pepper"
image_with_noise = image.copy()
image_with_noise[salt_and_pepper < noise_density / 2] = 0
image_with_noise[salt_and_pepper > 1 - noise_density / 2] = 255


# Εμφανίζουμε τις εικόνες στον ίδιο γράφο
plt.figure(figsize=(10, 5))

# Προβολή της αρχικής εικόνας
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Προβολή της εικόνας με τον θόρυβο "salt & pepper"
plt.subplot(1, 2, 2)
plt.imshow(image_with_noise, cmap='gray')
plt.title('Image with Salt & Pepper Noise')
plt.axis('off')

plt.show()

# C https://medium.com/@turgay2317/image-processing-in-python-with-opencv-blur-3e474fda6a52
# Εφαρμογή φίλτρου μέσης τιμής 3x3
filtered_image_3x3 = cv2.blur(image_with_noise, (3, 3))

# Εφαρμογή φίλτρου μέσης τιμής 5x5
filtered_image_5x5 = cv2.blur(image_with_noise, (5, 5))

# Εφαρμογή φίλτρου μέσης τιμής 7x7
filtered_image_7x7 = cv2.blur(image_with_noise, (7, 7))

# Εμφανίζουμε τις εικόνες σε ένα ενιαίο σχήμα
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