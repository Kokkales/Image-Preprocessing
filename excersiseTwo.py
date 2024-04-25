import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import cv2
from scipy.fft import fft2


# import image
try:
    toProcessImage = Image.open("../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp")
except OSError:
    raise ValueError("There was a problem with the input image.")

# -----------------------------------------------------------------------------------QUESTION 2a, 2b
# Υπολογίζουμε τον DFT
dft = np.fft.fft2(toProcessImage)

# Μετατρέπουμε το αποτέλεσμα σε κέντρο
dft_shift = np.fft.fftshift(dft)

# # Εμφανίζουμε το αποτέλεσμα - FASMA PLATOUS
# magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
# magnitude_spectrum = np.uint8(magnitude_spectrum)
# cv2.imshow('DFT of Image', magnitude_spectrum)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # FASMA PHASIS
# # Υπολογίζουμε τη φάση
# phase_spectrum = np.angle(dft_shift)
# cv2.imshow('Phase Spectrum', np.uint8(phase_spectrum))
# cv2.waitKey(0)
# Υπολογίζουμε τις διαστάσεις της εικόνας


# # ------------------------------------------------------------------------------------------------------------------- QUESTION 2c
# Load the image
toProcessImage = cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)

# Convert image to float32
toProcessImage = np.array(toProcessImage, dtype=np.float32)

# Calculate the DFT
dft = cv2.dft(toProcessImage, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the DFT result
dft_shift = np.fft.fftshift(dft)

# Get the dimensions of the image
rows, cols = toProcessImage.shape

# Create masks to keep only the 20% and 40% of the low frequencies
mask_20 = np.zeros((rows, cols), np.uint8)
mask_40 = np.zeros((rows, cols), np.uint8)

center_row, center_col = rows // 2, cols // 2
radius_20 = int(0.2 * min(rows, cols))
radius_40 = int(0.4 * min(rows, cols))
mask_20[center_row - radius_20:center_row + radius_20, center_col - radius_20:center_col + radius_20] = 1
mask_40[center_row - radius_40:center_row + radius_40, center_col - radius_40:center_col + radius_40] = 1

# Apply the masks to the DFT
dft_shift_20 = dft_shift * mask_20[:, :, np.newaxis]
dft_shift_40 = dft_shift * mask_40[:, :, np.newaxis]

# Calculate the inverse DFT
idft_20 = np.fft.ifftshift(dft_shift_20)
idft_20 = cv2.idft(idft_20)
idft_20 = cv2.magnitude(idft_20[:, :, 0], idft_20[:, :, 1])

idft_40 = np.fft.ifftshift(dft_shift_40)
idft_40 = cv2.idft(idft_40)
idft_40 = cv2.magnitude(idft_40[:, :, 0], idft_40[:, :, 1])

# Display the reconstructed images
cv2.imshow('Reconstructed Image (20% of low frequencies)', np.uint8(idft_20))
cv2.imshow('Reconstructed Image (40% of low frequencies)', np.uint8(idft_40))
cv2.waitKey(0)

# import cv2
# import numpy as np

# def low_pass_filter(dft, percentage):
#   """Filters the DFT by keeping only a percentage of low frequencies."""
#   rows, cols = dft.shape[:2]
#   center_row, center_col = int(rows / 2), int(cols / 2)
#   mask = np.zeros((rows, cols), dtype=np.float32)

#   # Create circular mask for low frequencies
#   radius = int(min(center_row, center_col) * percentage)
#   cv2.circle(mask, (center_col, center_row), radius, (1, 1, 1), -1)

#   # Apply mask to DFT (set high frequencies to zero)
#   mask = mask[..., np.newaxis]
#   dft_filtered = dft * mask

# #   real_dft, imag_dft = cv2.split(dft)
# #   dft_filtered_real = real_dft * mask
# #   dft_filtered_imag = imag_dft * mask
# #   dft_filtered = cv2.merge((dft_filtered_real, dft_filtered_imag))

#   return dft_filtered

# # Load the image (replace "barbara.bmp" with your path)
# image = cv2.imread("../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp", cv2.IMREAD_GRAYSCALE)

# # Perform 2D DFT using cv2.dft
# dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

# # Keep 20% of low frequencies
# percentage = 0.2
# dft_filtered = low_pass_filter(dft, percentage)

# # Perform inverse DFT (iFFT) for reconstruction
# reconstructed_image = cv2.idft(dft_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

# # Normalize reconstructed image (optional)
# reconstructed_image = cv2.normalize(reconstructed_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# # Visualization
# cv2.imshow("Original Image", image)
# cv2.imshow("Reconstructed Image (20% Low Frequencies)", reconstructed_image.astype(np.uint8))
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
