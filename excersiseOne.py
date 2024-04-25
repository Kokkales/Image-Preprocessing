import numpy as np
from PIL import Image

def umrq(image_array, levels):
    """
    Υλοποίηση UMRQ για εικόνες αποχρώσεων του γκρι.

    Args:
        image_array (ndarray): Η εικόνα αποχρώσεων του γκρι σε μορφή NumPy array.
        levels (int): Ο αριθμός των επιπέδων κβάντισης.

    Returns:
        ndarray: Η κβαντισμένη εικόνα.
    """
    # Υπολογισμός του εύρους τιμών
    min_value = np.min(image_array)
    max_value = np.max(image_array)
    print(min_value,max_value)

    # Υπολογισμός του βήματος κβάντισης
    step = (max_value - min_value) / (levels - 1)

    # Δημιουργία πίνακα κβαντισμένων τιμών
    quantized_image = np.zeros_like(image_array)

    # Κβάντιση κάθε pixel
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            pixel_value = image_array[i, j]

            # Υπολογισμός του δείκτη στάθμης κβάντισης
            level = int((pixel_value - min_value) // step)

            # Κβάντιση pixel
            quantized_image[i, j] = min_value + level * step

    return quantized_image

# Load image
try:
    image = Image.open("../DIP-project-1/DIP-project-1/images-project-1/barbara.bmp")
except OSError:
    print("Error opening image. Check file path or format.")
    exit()

# Convert to grayscale if needed (assuming barbara.bmp is grayscale)
grayscale_image = image.convert('L')

# Convert to NumPy array
image_array = np.array(grayscale_image)
print(image_array.shape)

# Κβάντιση με UMRQ
quantized_image_array_8 = umrq(image_array, 8)
quantized_image_array_12 = umrq(image_array, 12)
quantized_image_array_16 = umrq(image_array, 16)
quantized_image_array_20 = umrq(image_array, 20)

# Μετατροπή σε εικόνες Pillow
quantized_image_8 = Image.fromarray(quantized_image_array_8)
quantized_image_12 = Image.fromarray(quantized_image_array_12)
quantized_image_16 = Image.fromarray(quantized_image_array_16)
quantized_image_20 = Image.fromarray(quantized_image_array_20)

# Αποθήκευση κβαντισμένων εικόνων
quantized_image_8.save("quantized_barbara_8_pillow.bmp")
quantized_image_12.save("quantized_barbara_12_pillow.bmp")
quantized_image_16.save("quantized_barbara_16_pillow.bmp")
quantized_image_20.save("quantized_barbara_20_pillow.bmp")
# print(caseResults)
# quantum image
# qImageEight, qEight_transformations= quantImg(imageArray, 8)
# qImageTwelve, qTwelve_transformations = quantImg(imageArray, 12)
# qImageSixteen, qSixteen_transformations  = quantImg(imageArray, 16)
# qImageTwenty, qTwenty_transformations = quantImg(imageArray, 20)
# # qImageFourty = quantImg(imageArray, 4)

# # convert from numpy back to pillow
# qImageEight = Image.fromarray(qImageEight)
# qImageTwelve = Image.fromarray(qImageTwelve)
# qImageSixteen = Image.fromarray(qImageSixteen)
# qImageTwenty= Image.fromarray(qImageTwenty)
# # qImageFourty= Image.fromarray(qImageFourty)


# # save new image
# qImageEight.save("./results/qBarbara_8.bmp")
# qImageTwelve.save("./results/qBarbara_12.bmp")
# qImageSixteen.save("./results/qBarbara_16.bmp")
# qImageTwenty.save("./results/qBarbara_20.bmp")
# # qImageFourty.save("./results/qBarbara_40.bmp")
