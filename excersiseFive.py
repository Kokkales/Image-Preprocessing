import cv2
import numpy as np
import matplotlib.pyplot as plt

# Διάβασμα της εικόνας
image = cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Δημιουργία του Γκαουσιανού θορύβου
mean = 0
variance = 0.01
sigma = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, sigma, image.shape)
gaussian_noise = gaussian_noise.reshape(image.shape).astype(np.uint8)

# Προσθήκη θορύβου στην εικόνα
noisy_image = cv2.add(image, gaussian_noise)

# Εμφάνιση των εικόνων πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.show()
