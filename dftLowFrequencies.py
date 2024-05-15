import cv2
import numpy as np
import matplotlib.pyplot as plt


# B
def plotMagnitudeAndSpectrum(dftShifted, image):
    # magnitude and phase spectrum
    magnitudeSpectrum = 20 * np.log(cv2.magnitude(dftShifted[:, :, 0], dftShifted[:, :, 1]))
    phaseSpectrum = np.angle(dftShifted)
    # print(phaseSpectrum.shape)
    # Split the image into real and imaginary parts
    realPart = phaseSpectrum[:,:,0]
    imaginaryPart = phaseSpectrum[:,:,1]
    # Calculate phase angle
    phaseAngle = np.arctan2(imaginaryPart, realPart)
    phaseAngleNormalized = (phaseAngle + np.pi) / (2 * np.pi) # normalize
    phaseImage = (phaseAngleNormalized * 255).astype(np.uint8)

    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(magnitudeSpectrum ),plt.title('Magnitude Spectrum') # can plot it also in cmap='gray'
    plt.subplot(133), plt.imshow(phaseImage),plt.title('Phase Spectrum') # can plot it also in cmap='gray'
    plt.savefig("./results_two/spectrum_images.png")
    plt.show()

# Γ
def lowFreqReconstructor(dft, image):
    # filter: keep only low-pass frequencies (20% and 40%)
    filteredDft20 = np.zeros_like(dft)
    filteredDft20[:int(0.2 * dft.shape[0]), :int(0.2 * dft.shape[1])] = dft[:int(0.2 * dft.shape[0]), :int(0.2 * dft.shape[1])]

    fileteredDft40 = np.zeros_like(dft)
    fileteredDft40[:int(0.4 * dft.shape[0]), :int(0.4 * dft.shape[1])] = dft[:int(0.4 * dft.shape[0]), :int(0.4 * dft.shape[1])]

    # idft
    reconstructedImage20 = cv2.idft(filteredDft20, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    reconstructedImage40 = cv2.idft(fileteredDft40, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(reconstructedImage20, cmap='gray'), plt.title('Rbuilded (20%)')
    plt.subplot(133), plt.imshow(reconstructedImage40, cmap='gray'), plt.title('Rebuilded (40%)')
    plt.savefig('./results_two/rebuilded_images.png')
    plt.show()
    calculateMSE(image, reconstructedImage20, 20)
    calculateMSE(image, reconstructedImage40, 40)

# Δ
def calculateMSE(image, newImage, level):
    MSE = np.mean((image.astype(np.float32) - newImage.astype(np.float32)) ** 2)
    print(f"MSE of {level}: {MSE}")

if __name__=='__main__':
    # import image
    image = cv2.imread('images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)
    # A - dft
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft) # center pectrum
    plotMagnitudeAndSpectrum(dftShift, image)
    lowFreqReconstructor(dft, image)
