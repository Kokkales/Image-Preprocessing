import numpy as np
import cv2

def dct(image):
    # Διάσταση της εικόνας
    M, N = image.shape

    # Δημιουργία ενός κενού πίνακα για το αποτέλεσμα
    result = np.zeros((M, N), dtype=np.float32)

    # Υπολογισμός του DCT
    for u in range(M):
        print(u)
        for v in range(N):
            cu = 1 if u == 0 else np.sqrt(2)
            cv = 1 if v == 0 else np.sqrt(2)
            sum_val = 0
            for x in range(M):
                for y in range(N):
                    sum_val += image[x, y] * np.cos((2*x + 1) * u * np.pi / (2 * M)) * np.cos((2*y + 1) * v * np.pi / (2 * N))
            result[u, v] = cu * cv * sum_val / np.sqrt(M * N)

    return result

def idct(dct_coefficients):
    # Διάσταση της εικόνας
    M, N = dct_coefficients.shape

    # Δημιουργία ενός κενού πίνακα για το αποτέλεσμα
    result = np.zeros((M, N), dtype=np.float32)

    # Υπολογισμός του IDCT
    for x in range(M):
        print(x)
        for y in range(N):
            sum_val = 0
            for u in range(M):
                for v in range(N):
                    cu = 1 if u == 0 else np.sqrt(2)
                    cv = 1 if v == 0 else np.sqrt(2)
                    sum_val += cu * cv * dct_coefficients[u, v] * np.cos((2*x + 1) * u * np.pi / (2 * M)) * np.cos((2*y + 1) * v * np.pi / (2 * N))
            result[x, y] = sum_val / np.sqrt(M * N)

    return result

# Διάβασμα της εικόνας grayscale
image = cv2.imread('../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)

# Υπολογισμός του DCT
dct_result = dct(image)

# Εκτύπωση του φάσματος DCT
print("DCT spectrum: ", dct_result)
# print(dct_result)

# Υπολογισμός του IDCT
idct_result = idct(dct_result)

# Εκτύπωση της ανακτημένης εικόνας
cv2.imshow('Recovered Image', idct_result.astype(np.uint8))
cv2.waitKey(0)

mse = np.mean((image - idct_result) ** 2)
print("MSE: ", mse)