import cv2
import numpy as np
from matplotlib import pyplot as plt

# Φόρτωση της εικόνας
image = cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)

# Εκτέλεση του DFT
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Παρουσίαση του φάσματος στη συχνοτική περιοχή
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# Compute the phase spectrum
phase_spectrum = np.angle(dft_shift)
print(phase_spectrum.shape)
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Εικόνα του χώρου'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Φάσμα των συχνοτήτων'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(phase_spectrum)
# plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
# plt.axis('off')
# plt.tight_layout()
plt.show()
# cv2.imshow('Phase Spectrum', np.uint8(phase_spectrum))
# cv2.waitKey(0)

# TODO FIX THIS
# Split the image into real and imaginary parts
real_part = phase_spectrum[:,:,0]
imaginary_part = phase_spectrum[:,:,1]
# Calculate phase angle
phase_angle = np.arctan2(imaginary_part, real_part)
# Normalize phase angle to [0, 1] for visualization
phase_angle_normalized = (phase_angle + np.pi) / (2 * np.pi)
# Convert to uint8 for visualization
phase_image_uint8 = (phase_angle_normalized * 255).astype(np.uint8)
# Display the phase image
# cv2.imshow('Phase Image', phase_image_uint8)
# cv2.waitKey(0)
plt.imshow(phase_image_uint8, cmap='gray')  # Assuming grayscale visualization
plt.axis('off')  # Hide axes
plt.title('Phase Image')
plt.show()

# C
# Υπολογισμός του μεγέθους του DFT
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Εφαρμογή της μάσκας για τη διατήρηση του 20% και 40% των συντελεστών
mask_20 = np.zeros((rows, cols, 2), np.uint8)
mask_40 = np.zeros((rows, cols, 2), np.uint8)

mask_20[crow - int(0.2 * crow):crow + int(0.2 * crow), ccol - int(0.2 * ccol):ccol + int(0.2 * ccol)] = 1
mask_40[crow - int(0.4 * crow):crow + int(0.4 * crow), ccol - int(0.4 * ccol):ccol + int(0.4 * ccol)] = 1

# Εφαρμογή του φίλτρου
fshift_20 = dft_shift * mask_20
fshift_40 = dft_shift * mask_40

# Αντίστροφος μετασχηματισμός Fourier
f_ishift_20 = np.fft.ifftshift(fshift_20)
f_ishift_40 = np.fft.ifftshift(fshift_40)

img_back_20 = cv2.idft(f_ishift_20)
img_back_40 = cv2.idft(f_ishift_40)

img_back_20 = cv2.magnitude(img_back_20[:, :, 0], img_back_20[:, :, 1])
img_back_40 = cv2.magnitude(img_back_40[:, :, 0], img_back_40[:, :, 1])

# Παρουσίαση των ανακατασκευασμένων εικόνων
plt.subplot(121), plt.imshow(img_back_20, cmap='gray')
plt.title('Ανακατασκευή με 20% συντελεστές'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back_40, cmap='gray')
plt.title('Ανακατασκευή με 40% συντελεστές'), plt.xticks([]), plt.yticks([])
plt.show()

# D MSE error
# Calculate MSE between original and reconstructed images
mse_20 = np.mean((image.astype(float) - img_back_20.astype(float)) ** 2)
mse_40 = np.mean((image.astype(float) - img_back_40.astype(float)) ** 2)

print("MSE for 20% coefficients retained:", mse_20)
print("MSE for 40% coefficients retained:", mse_40)