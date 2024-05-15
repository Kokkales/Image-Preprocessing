import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    image = cv2.imread('./images-project-1/lenna.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]

    # Α
    # noise density "salt & pepper"
    noiseDensity = 0.05
    # create noise
    saltAndPepperNoise = np.random.rand(height, width)
    # apply the noise
    noisyImage = image.copy()
    noisyImage[saltAndPepperNoise < noiseDensity / 2] = 0
    noisyImage[saltAndPepperNoise > 1 - noiseDensity / 2] = 255

    plt.figure(figsize=(10, 5)), plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
    plt.subplot(122), plt.imshow(noisyImage, cmap='gray'), plt.title('Image with Salt & Pepper Noise'), plt.axis('off')
    plt.savefig('./results_four/image_with_noise.png')
    plt.show()

    # B
    # C https://medium.com/@turgay2317/image-processing-in-python-with-opencv-blur-3e474fda6a52
    # mean filter 3x3
    filteredImage3x3 = cv2.blur(noisyImage, (3, 3))
    # mean filter 5x5
    filteredImage5x5 = cv2.blur(noisyImage, (5, 5))
    # mean filter 7x7
    filteredImage7x7 = cv2.blur(noisyImage, (7, 7))

    # Γ
    plt.figure(figsize=(12, 8))
    plt.subplot(221), plt.imshow(noisyImage, cmap='gray'), plt.title('Image with Salt & Pepper Noise'), plt.axis('off')
    plt.subplot(222), plt.imshow(filteredImage3x3, cmap='gray'), plt.title('Filtered Image (3x3)'), plt.axis('off')
    plt.subplot(223), plt.imshow(filteredImage5x5, cmap='gray'), plt.title('Filtered Image (5x5)'), plt.axis('off')
    plt.subplot(224), plt.imshow(filteredImage7x7, cmap='gray'), plt.title('Filtered Image (7x7)'), plt.axis('off')
    plt.savefig('./results_four/images_after_filters.png')
    plt.show()