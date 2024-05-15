import cv2
import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt


def butterworth_filter(image, f_cutoff, n):
  """
  Applies a Butterworth filter to an image in the frequency domain.

  Args:
      image: Grayscale image as a NumPy array.
      f_cutoff: Cut-off frequency in Hz.
      n: Order of the Butterworth filter.

  Returns:
      Filtered image as a NumPy array, or None if an error occurs.
  """
  try:
    # Get image dimensions
    rows, cols = image.shape

    # Sample rate (assuming square image)
    fs = 1 / (rows / 256)  # Assuming image size is 256x256, adjust if different

    # Normalize cut-off frequency
    d0 = f_cutoff / (fs / 2)  # Normalize based on Nyquist frequency

    # Perform FFT to get image spectrum
    dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create frequency mask for Butterworth filter
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    d = np.sqrt(u**2 + v**2)
    _, H = butter(n, d0, btype='low')  # Generate Butterworth filter response

    # Apply filter in frequency domain
    dft_shift *= H

    # Shift back and perform inverse FFT to get filtered image
    filtered_dft = np.fft.ifftshift(dft_shift)
    filtered_image = cv2.idft(filtered_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return filtered_image.real.astype(np.uint8)
  except (ValueError, ZeroDivisionError) as e:
    print(f"Error applying filter: {e}")
    return None


def main():
  # Load image (replace 'lenna.bmp' with your actual image path)
  image = cv2.imread("./images-project-1/lenna.bmp", 0)
  print('ok')

  # Check if image loaded successfully
  if image is None:
    print("Error: Could not load image")
    return

  # Add Gaussian noise
  gauss_noise = np.random.normal(loc=0, scale=1, size=image.shape).astype(np.uint8)
  gn_img = cv2.add(image, gauss_noise)

  # Display original, noise, and combined image
  plt.subplot(131), plt.imshow(image, cmap='gray'), plt.axis("off"), plt.title("Original")
  plt.subplot(132), plt.imshow(gauss_noise, cmap='gray'), plt.axis("off"), plt.title("Gaussian Noise")
  plt.subplot(133), plt.imshow(gn_img, cmap='gray'), plt.axis("off"), plt.title("Combined")
  plt.savefig('./results-five/gauss_noise_image.jpg')
  plt.show()

  # Define cut-off frequency (adjust as needed)
  f_cutoff = 10  # Adjust this value based on your image and desired filtering

  # Apply Butterworth filters of different orders
  filtered_image_3 = butterworth_filter(gn_img.copy(), f_cutoff, 3)
  filtered_image_5 = butterworth_filter(gn_img.copy(), f_cutoff, 5)
  filtered_image_7 = butterworth_filter(gn_img.copy(), f_cutoff, 7)

  # Check if filtering was successful
  if filtered_image_3 is None or filtered_image_5 is None or filtered_image_7 is None:
    print("Error: An error occurred during filtering. See console for details.")
    return

  # Display results
  plt.subplot(221), plt.imshow(gn_img, cmap='gray'), plt.axis("off"), plt.title("Noisy Image")
  plt.subplot(222), plt.imshow(filtered_image_3, cmap='gray'), plt.axis("off"), plt.title("Filtered (3rd)")
  plt.subplot(222), plt.imshow(filtered_image_5, cmap='gray'), plt.axis("off"), plt.title("Filtered (5th)")
  plt.subplot(222), plt.imshow(filtered_image_7, cmap='gray'), plt.axis("off"), plt.title("Filtered (7th)")
  plt.show()


main()