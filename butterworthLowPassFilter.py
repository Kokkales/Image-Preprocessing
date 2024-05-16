# https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python
import numpy as np
import matplotlib.pyplot as plt
import cv2

image=cv2.imread("./images-project-1/lenna.bmp",0)

# Add noise
gaussNoise=np.random.normal(loc=0, scale=1, size=image.shape).astype(np.uint8)
gnImg=cv2.add(image,gaussNoise)

plt.subplot(131), plt.imshow(image,cmap='gray'), plt.axis("off"), plt.title("Original")
plt.subplot(132), plt.imshow(gaussNoise,cmap='gray'), plt.axis("off"), plt.title("Gaussian Noise")
plt.subplot(133), plt.imshow(gnImg,cmap='gray'), plt.axis("off"), plt.title("Combined")
plt.savefig('./results-five/gauss_noise_image.jpg')
plt.show()

def butterworthFilter(shape, cutoffFreq, order):
    x = np.linspace(-0.5, 0.5, shape[0])
    y = np.linspace(-0.5, 0.5, shape[1])
    xx, yy = np.meshgrid(x, y) # spatial frequencies
    radius = np.sqrt(xx**2 + yy**2) # eucledian distance of each point
    filterMask = 1 / (1 + (radius / cutoffFreq)**(2 * order))  # compute buterworth filter mas
    return filterMask

def applyFilter(image, filterMask):
    fftImage = np.fft.fftshift(np.fft.fft2(image))
    filteredImage = fftImage * filterMask
    filteredImage = np.fft.ifft2(np.fft.ifftshift(filteredImage)).real
    return filteredImage

image = gnImg
# Define cutoff frequencies and filter orders
orders = [3, 5, 7]
cutoffFrequencies = [0.5, 0.07, 0.05]
# Plot original image
plt.figure(figsize=(12, 4))
plt.subplot(1, len(orders) + 1, 1)
plt.imshow(gnImg, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Process the image for each cutoff frequency and order
for i, order in enumerate(orders):
    cutoffFreq = cutoffFrequencies[i]
    filterMask = butterworthFilter(gnImg.shape, cutoffFreq, order)
    filteredImage = applyFilter(gnImg, filterMask)

    plt.subplot(1, len(orders) + 1, i+2), plt.imshow(filteredImage, cmap='gray'), plt.title('Order: {}'.format(order, cutoffFreq)),plt.axis('off')
plt.savefig('./results-five/filtered_images.jpg')
plt.show()