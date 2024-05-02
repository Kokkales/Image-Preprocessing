import numpy as np
import cv2
import matplotlib.pyplot as plt

def dct(image):
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            if u==0:
                au= 1 / np.sqrt(M)
            else:
                au = np.sqrt(2/M)
            if v==0:
                av= 1 / np.sqrt(N)
            else:
                av = np.sqrt(2/N)
            sum = 0
            for m in range(M):
                for n in range(N):
                    cosU= np.cos(np.pi * (2 * m + 1) * u / (2 * M))
                    cosV= np.cos(np.pi * (2 * n + 1) * v / (2 * N))
                    sum+= image[m,n]*cosU*cosV
            result[u,v]=au*av*sum

    return result

def idct(coefficients):
    M, N = coefficients.shape
    result = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            sum = 0
            for u in range(M):
                for v in range(N):
                    if u == 0:
                        au = 1 / np.sqrt(M)
                    else:
                        au = np.sqrt(2 / M)
                    if v == 0:
                        av = 1 / np.sqrt(N)
                    else:
                        av = np.sqrt(2 / N)
                    cosU = np.cos(np.pi * (2 * m + 1) * u / (2 * M))
                    cosV = np.cos(np.pi * (2 * n + 1) * v / (2 * N))
                    sum += au * av * coefficients[u, v] * cosU * cosV
            result[m, n] = sum

    return result

# import image
image = cv2.imread('./images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)
# Calculate dct
dct_result = dct(image)
# Calculate idct
idct_result = idct(dct_result)
# dct magnitude spectrum
dct_spectrum = np.log(1 + np.abs(dct_result))

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(idct_result.astype(np.uint8), cmap='gray')
plt.title('Reconstructed Image')
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(dct_spectrum)
plt.title('DCT Spectrum')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig("./results_three/dct_image.png")
plt.show()