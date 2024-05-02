
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

# Β. Εφαρμογή φίλτρων Butterworth στο πεδίο της συχνότητας

# Compute the Fourier transform of the noisy image
f = np.fft.fft2(gn_img)
fshift = np.fft.fftshift(f)

# Define the cutoff frequency for the filters
cutoff_freq_3 = 0.1
cutoff_freq_5 = 0.05
cutoff_freq_7 = 0.03

# Initialize lists to store filtered images
filtered_images = []

# Apply Butterworth low-pass filters of 3rd, 5th, and 7th order
for order in [3, 5, 7]:
    # Define the Butterworth low-pass filter
    butterworth_filter = 1 / (1 + ((np.sqrt((np.fft.fftfreq(gn_img.shape[0]) ** 2)[:, np.newaxis] +
                                             (np.fft.fftfreq(gn_img.shape[1]) ** 2)[np.newaxis, :])) / cutoff_freq_3) ** (2 * order))

    # Apply the filter in the frequency domain
    filtered_fshift = fshift * butterworth_filter

    # Compute the inverse Fourier transform
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fshift)))

    # Append the filtered image to the list
    filtered_images.append(filtered_image)


# Γ. Εμφάνιση αρχικής και φιλτραρισμένης εικόνας

# Display original and filtered images together
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, len(filtered_images)+1, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Filtered Images
for i, filtered_image in enumerate(filtered_images, start=1):
    plt.subplot(1, len(filtered_images)+1, i+1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Filtered (Order {2*i+1})')
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
