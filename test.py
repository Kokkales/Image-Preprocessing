import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# # Read the image
# img = cv2.imread("./images-project-1/lenna.bmp", cv2.IMREAD_GRAYSCALE)

# # Add Gaussian noise
# mean = 0
# std_dev = 10
# noise = np.random.normal(mean, std_dev, img.shape).astype(np.float32)
# noisy_img = img + noise

# # Apply Butterworth low-pass filters
# orders = [3, 5, 7]
# cutoff_freq = 0.1  # Adjust cutoff frequency as needed

# filtered_images = []

# for order in orders:
#     # Create Butterworth low-pass filter
#     b, a = butter(order, cutoff_freq, btype='low')

#     # Apply filter to the frequency domain representation
#     f = np.fft.fft2(noisy_img)
#     fshift = np.fft.fftshift(f)
#     filtered_fshift = filtfilt(b, a, fshift)

#     # Perform inverse Fourier transform to obtain the filtered image in the spatial domain
#     filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_fshift)).real

#     # Clip the filtered image
#     filtered_img = np.clip(filtered_img, 0, 255)

#     # Store the filtered image
#     filtered_images.append(filtered_img)

# # Plotting
# plt.figure(figsize=(15, 5))

# # Plot original image and noisy images
# plt.subplot(1, len(orders) + 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, len(orders) + 2, 2)
# plt.imshow(noisy_img, cmap='gray')
# plt.title('Noisy Image')
# plt.axis('off')

# # Plot filtered images
# for i, order in enumerate(orders):
#     plt.subplot(1, len(orders) + 2, i + 3)
#     plt.imshow(filtered_images[i], cmap='gray')
#     plt.title(f'Filtered (Order {order})')
#     plt.axis('off')

# plt.show()

# import cv2

# # Load the image
# image = cv2.imread('./images-project-1/barbara.bmp')

# # Method 1: Resize the image to a smaller size
# # Define the new dimensions
# new_width = 40
# new_height = 40
# # Resize the image
# resized_image = cv2.resize(image, (new_width, new_height))


# # Display the original and processed images
# cv2.imshow('Original Image', image)
# cv2.imshow('Resized Image', resized_image)
# # cv2.imshow('Compressed Image', decompressed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the processed images
# # cv2.imwrite('resized_image.jpg', resized_image)
# cv2.imwrite('barbara_compressed.bmp', resized_image)

import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("./images-project-1/lenna.bmp",0)

gauss_noise=np.zeros(img.shape,dtype=np.uint8)
cv2.randn(gauss_noise,128,20)
gauss_noise=(gauss_noise*0.1).astype(np.uint8)

gn_img=cv2.add(img,gauss_noise)

fig=plt.figure(dpi=300)

fig.add_subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.axis("off")
plt.title("Original")

fig.add_subplot(1,3,2)
plt.imshow(gauss_noise,cmap='gray')
plt.axis("off")
plt.title("Gaussian Noise")

fig.add_subplot(1,3,3)
plt.imshow(gn_img,cmap='gray')
plt.axis("off")
plt.title("Combined")
plt.show()

# # Load your image
# # Replace 'image_path' with the path to your image
# original_image = plt.imread('./images-project-1/lenna.bmp')


# plt.figure(figsize=(10, 5))

# # Plot original image
# plt.subplot(1, 2, 1)
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# # Plot noisy image
# plt.subplot(1, 2, 2)
# plt.imshow(noisy_image, cmap='gray')
# plt.title('Noisy Image')
# plt.axis('off')

# plt.show()
