# import matplotlib.pyplot as plt
# import numpy as np

# def plot_data_with_step(data, step):
#   """
#   Plots the data points with a specified step size on the x-axis.

#   Args:
#       data: NumPy array containing the data points.
#       step: The step size for the x-axis.
#   """
#   x = np.arange(0, max(data)+step, step)  # Create x-axis with step

#   # Plot the data with markers
#   plt.plot(x, data, marker='o', linestyle='-')

#   # Set labels and title
#   plt.xlabel('X-axis (step size {})'.format(step))
#   plt.ylabel('Y-axis')
#   plt.title('Data Points with Step Size {}'.format(step))

#   # Show the plot with grid
#   plt.grid(True)
#   plt.show()

# # Example usage
# data = np.array([0, 30, 61, 91, 122, 153, 183, 214, 245])
# step_size = 30

# plot_data_with_step(data, step_size)

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img=cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# dft = np.fft.fft2(img)
# dft_shift = np.fft.fftshift(dft)
# phase_spectrum = np.angle(dft_shift)

# ax1 = plt.subplot(1,2,1)
# ax1.imshow(img, cmap='gray')

# ax2 = plt.subplot(1,2,2)
# ax2.imshow(phase_spectrum, cmap='gray')

# plt.show()

import cv2
import numpy as np

# def low_pass_filter(image, percentage, axis=2):
#   """
#   Extracts low frequencies from an image using frequency domain filtering.

#   Args:
#       image: cv image (numpy array)
#       percentage: percentage of low frequencies to keep (0.0 to 1.0)
#       axis: axis along which to apply the filter (default: 2 for color images)

#   Returns:
#       image containing only the selected percentage of low-frequency components
#   """
#   # Convert image to grayscale if colored
#   if len(image.shape) == 3:
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#   # Get image dimensions and calculate cutoff radius
#   rows, cols = image.shape[:2]  # Consider only the first two dimensions
#   radius = int(percentage * min(rows, cols) / 2)

#   # Apply FFT
#   dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
#   dft_shift = np.fft.fftshift(dft, axes=axis)  # Shift along specified axis

#   # Create low-pass filter mask
#   mask = np.zeros((rows, cols), np.uint8)
#   cv2.circle(mask, (rows // 2, cols // 2), radius, 255, -1)

#   # Apply mask, perform IFFT, and return filtered image
#   def apply_filter(shifted_dft, mask_copy):
#     mask_expanded = np.repeat(mask_copy[..., np.newaxis],2)

#     # Apply mask and perform IFFT
#     filtered_dft = shifted_dft * mask_expanded
#     # filtered_dft = shifted_dft * mask_copy
#     filtered_image = np.real(np.fft.ifftshift(filtered_dft, axes=axis))
#     return filtered_image.astype(np.uint8)

#   # Perform filtering for 20% and 40% successively
#   filtered_image_20 = apply_filter(dft_shift.copy(), mask.copy())
#   filtered_image_40 = apply_filter(dft_shift.copy(), np.repeat(mask[..., np.newaxis], 2, axis=-1)[:rows, :cols])  # Expand mask for 40%

#   return filtered_image_20, filtered_image_40  # Return both filtered images

# # Load your image
# image = cv2.imread("../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp")

# # Extract low frequencies with 20% and 40% thresholds
# filtered_image_20, filtered_image_40 = low_pass_filter(image.copy(), 0.2)
# filtered_image_40_color = cv2.cvtColor(filtered_image_40, cv2.COLOR_GRAY2BGR)  # Convert back to color for display

# # Display original, 20% filtered, and 40% filtered images
# cv2.imshow("Original Image", image)
# cv2.imshow("20% Low Frequencies", filtered_image_20)
# cv2.imshow("40% Low Frequencies", filtered_image_40_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# test = np.array([2,3,4],[5,6,7])
# test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# # print(array_example)
# print(0.5*np.min(test))
# print(test.shape)

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
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Εικόνα του χώρου'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Φάσμα των συχνοτήτων'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

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
