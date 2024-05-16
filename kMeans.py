import numpy as np
import cv2
import matplotlib.pyplot as plt

# Α
def kMeans(image, k, max_iterations=50):
    # 1. set k random pixels
    centroids = image[np.random.choice(image.shape[0], k, replace=False)]
    for i in range(max_iterations):
        # 2. calculate the distance from all the pixels from all the centroids
        distances = np.zeros((image.shape[0], k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(image - centroids[j], axis=1)
        # 3. check for each pixel distance in which centroid is closer and label it
        labels = np.argmin(distances, axis=1)
        # 4. repeat this till max iterations are 50 or new_centroid = centroid
        for j in range(k):
            newCentroids = np.array(image[labels == j].mean(axis=0))
        if np.all(centroids == newCentroids):
            break
    return centroids, labels

# place the colors according to kmeans labeling
def placeColors(image, centroids, labels):
    newImage = centroids[labels]
    return newImage.reshape(image.shape)

def imageSegmentation(defaultImage, kValues):
    fig, axes = plt.subplots(1, len(kValues), figsize=(15, 5))
    for i, k in enumerate(kValues):
        centroids, labels = kMeans(defaultImage.reshape(-1, 3), k, 50)
        newΙmage = placeColors(defaultImage, centroids, labels)
        # Convert the image to grayscale before saving
        grayΙmage = cv2.cvtColor(newΙmage.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Plot the segmentation mask
        axes[i].imshow(grayΙmage, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"K={k}")
    plt.tight_layout()
    plt.savefig('./results_six/image_segmeted.jpg')
    plt.show()

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("./images-project-1/butterfly.jpg"), cv2.COLOR_BGR2RGB)
    print(image.shape)
    kValues = [5, 10, 15]
    imageSegmentation(image, kValues)
