import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Read the image
img = cv2.imread("./images-project-1/lenna.bmp", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
mean = 0
std_dev = 10
noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)
noisy_img = img + noise

# Apply Butterworth low-pass filters
orders = [3, 5, 7]
cutoff_freq = 0.1  # Adjust cutoff frequency as needed

filtered_images = []

for order in orders:
    # Create Butterworth low-pass filter
    b, a = butter(order, cutoff_freq, btype='low')

    # Apply filter to the frequency domain representation
    f = np.fft.fft2(noisy_img)
    fshift = np.fft.fftshift(f)
    filtered_fshift = filtfilt(b, a, fshift)

    # Perform inverse Fourier transform to obtain the filtered image in the spatial domain
    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_fshift)).real

    # Clip the filtered image
    filtered_img = np.clip(filtered_img, 0, 255)

    # Store the filtered image
    filtered_images.append(filtered_img)

# Plotting
plt.figure(figsize=(15, 5))

# Plot original image and noisy images
plt.subplot(1, len(orders) + 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, len(orders) + 2, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

# Plot filtered images
for i, order in enumerate(orders):
    plt.subplot(1, len(orders) + 2, i + 3)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title(f'Filtered (Order {order})')
    plt.axis('off')

plt.show()
