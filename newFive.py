# https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python
import numpy as np
import matplotlib.pyplot as plt
import cv2

image=cv2.imread("./images-project-1/lenna.bmp",0)

# Add noise
gauss_noise=np.random.normal(loc=0, scale=1, size=image.shape).astype(np.uint8)
print(gauss_noise)
gn_img=cv2.add(image,gauss_noise)

plt.subplot(131), plt.imshow(image,cmap='gray'), plt.axis("off"), plt.title("Original")
plt.subplot(132), plt.imshow(gauss_noise,cmap='gray'), plt.axis("off"), plt.title("Gaussian Noise")
plt.subplot(133), plt.imshow(gn_img,cmap='gray'), plt.axis("off"), plt.title("Combined")
plt.savefig('./results-five/gauss_noise_image.jpg')
plt.show()

def butterworth_filter(shape, cutoff_freq, order):
    x = np.linspace(-0.5, 0.5, shape[0])
    y = np.linspace(-0.5, 0.5, shape[1])
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    filter_mask = 1 / (1 + (radius / cutoff_freq)**(2 * order))
    return filter_mask

def apply_filter(image, filter_mask):
    fft_image = np.fft.fftshift(np.fft.fft2(image))
    filtered_image = fft_image * filter_mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_image)).real
    return filtered_image

image = gn_img
# Define cutoff frequencies and filter orders
orders = [3, 5, 7]
cutoff_frequencies = [0.5, 0.07, 0.05]
# Plot original image
plt.figure(figsize=(12, 4))
plt.subplot(1, len(orders) + 1, 1)
plt.imshow(gn_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Process the image for each cutoff frequency and order
for i, order in enumerate(orders):
    cutoff_freq = cutoff_frequencies[i]
    filter_mask = butterworth_filter(gn_img.shape, cutoff_freq, order)
    filtered_image = apply_filter(gn_img, filter_mask)

    plt.subplot(1, len(orders) + 1, i+2), plt.imshow(filtered_image, cmap='gray'), plt.title('Order: {}'.format(order, cutoff_freq)),plt.axis('off')
plt.savefig('./results-five/filtered_images.jpg')
plt.show()