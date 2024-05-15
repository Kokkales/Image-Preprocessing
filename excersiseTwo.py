import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# Shift the zero-frequency component to the center of the spectrum
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




# Filter by keeping only low-pass frequencies (20% and 40%)
dft_filtered_20 = np.zeros_like(dft)
dft_filtered_20[:int(0.2 * dft.shape[0]), :int(0.2 * dft.shape[1])] = dft[:int(0.2 * dft.shape[0]), :int(0.2 * dft.shape[1])]

dft_filtered_40 = np.zeros_like(dft)
dft_filtered_40[:int(0.4 * dft.shape[0]), :int(0.4 * dft.shape[1])] = dft[:int(0.4 * dft.shape[0]), :int(0.4 * dft.shape[1])]

# Perform inverse DFT using cv2.idft
image_reconstructed_20 = cv2.idft(dft_filtered_20, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
image_reconstructed_40 = cv2.idft(dft_filtered_40, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# Calculate MSE using only the real part of the difference
mse_20 = np.mean((image.astype(np.float32) - image_reconstructed_20) ** 2)
mse_40 = np.mean((image.astype(np.float32) - image_reconstructed_40) ** 2)

# Visualization (optional, using matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# You can uncomment these lines to show magnitude and phase
# plt.subplot(222), plt.imshow(magnitude, cmap='gray'), plt.title('Magnitude (Spectrum)')
# plt.subplot(223), plt.imshow(phase, cmap='hsv'), plt.title('Phase')
plt.subplot(222), plt.imshow(image_reconstructed_20, cmap='gray'), plt.title('Reconstruction (20%)')
plt.subplot(223), plt.imshow(image_reconstructed_40, cmap='gray'), plt.title('Reconstruction (40%)')
plt.show()

# Print MSE
print("MSE :", mse_20)
print("MSE :", mse_40)
