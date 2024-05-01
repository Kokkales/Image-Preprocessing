import cv2
import numpy as np
from matplotlib import pyplot as plt

# import image
image = cv2.imread('./images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)

# dft
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum )#, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

# Compute the phase spectrum
phase_spectrum = np.angle(dft_shift)
# print(phase_spectrum.shape)
# Split the image into real and imaginary parts
real_part = phase_spectrum[:,:,0]
imaginary_part = phase_spectrum[:,:,1]
# Calculate phase angle
phase_angle = np.arctan2(imaginary_part, real_part)
# Normalize phase angle to [0, 1] for visualization
phase_angle_normalized = (phase_angle + np.pi) / (2 * np.pi)
# Convert to uint8 for visualization
phase_image_uint8 = (phase_angle_normalized * 255).astype(np.uint8)

plt.subplot(133), plt.imshow(phase_image_uint8)#, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig("./results_two/spectrum_images.png")
plt.show()

# C
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# create masks
mask_20 = np.zeros((rows, cols, 2), np.uint8)
mask_40 = np.zeros((rows, cols, 2), np.uint8)

mask_20[crow - int(0.2 * crow):crow + int(0.2 * crow), ccol - int(0.2 * ccol):ccol + int(0.2 * ccol)] = 1
mask_40[crow - int(0.4 * crow):crow + int(0.4 * crow), ccol - int(0.4 * ccol):ccol + int(0.4 * ccol)] = 1

# apply filter
fshift_20 = dft_shift * mask_20
fshift_40 = dft_shift * mask_40

# Inverse dft
f_ishift_20 = np.fft.ifftshift(fshift_20)
f_ishift_40 = np.fft.ifftshift(fshift_40)

img_back_20 = cv2.idft(f_ishift_20)
img_back_40 = cv2.idft(f_ishift_40)

img_back_20 = cv2.magnitude(img_back_20[:, :, 0], img_back_20[:, :, 1])
img_back_40 = cv2.magnitude(img_back_40[:, :, 0], img_back_40[:, :, 1])

# Plot rebuild images
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back_20, cmap='gray')
plt.title('Rebuild with 20% '), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back_40, cmap='gray')
plt.title('Rebuild with 40% '), plt.xticks([]), plt.yticks([])
plt.savefig("./results_two/rebuilded_images.png")
plt.show()

# D MSE error
# Calculate MSE between original and reconstructed images
mse_20 = np.mean((image.astype(float) - img_back_20.astype(float)) ** 2)
mse_40 = np.mean((image.astype(float) - img_back_40.astype(float)) ** 2)

print("MSE for 20% coefficients retained:", mse_20)
print("MSE for 40% coefficients retained:", mse_40)