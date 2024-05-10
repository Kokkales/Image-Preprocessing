import numpy as np
import cv2
import matplotlib.pyplot as plt

# Μετασχηματισμός εικόνας στον χρωματικό χώρο RGB χρησιμοποιώντας την OpenCV
def image_to_rgb(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image_rgb[0][0][2])
    return image_rgb

# Υπολογισμός απόστασης μεταξύ δύο χρωμάτων στον χώρο RGB
def rgb_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

# # Υλοποίηση του αλγορίθμου K-means
# def k_means(image, k, max_iterations=50):
#     # Αρχικοποίηση τυχαίων κέντρων
#     centroids = image[np.random.choice(image.shape[0], k, replace=False)]

#     for _ in range(max_iterations):
#         # Υπολογισμός απόστασης κάθε pixel από τα κέντρα
#         distances = np.zeros((image.shape[0], k))
#         for i in range(k):
#             distances[:, i] = np.linalg.norm(image - centroids[i], axis=1)

#         # Κατηγοριοποίηση των pixels σε clusters ανάλογα με την ελάχιστη απόσταση
#         labels = np.argmin(distances, axis=1)

#         # Υπολογισμός νέων κέντρων
#         new_centroids = np.array([image[labels == i].mean(axis=0) for i in range(k)])

#         # Έλεγχος σύγκλισης
#         if np.all(centroids == new_centroids):
#             break

#         centroids = new_centroids

#     return centroids, labels

def k_means(image,k,max_iterations=50):
    # 1. set k random pixels
    print(image.shape)
    centroids=image[np.random.choice(image.shape[0], k, replace=False)]
    for i in range(max_iterations):
        # 2. calculate the distance from all the pixels from all the centroids
        distances=np.zeros((image.shape[0], k))
        print(image.shape[0])
        for j in range(k):
            distances[:, j] = np.linalg.norm(image - centroids[j], axis=1)

        # 3. check for each pixel distance in which centroid is closer and label it
        print(distances[0])
        labels = np.argmin(distances, axis=1)

        # repeat this till max iterations are 50 or new_centroid = centroid
        for j in range(k):
            new_centroids=np.array(image[labels == j].mean(axis=0))
        if np.all(centroids == new_centroids):
            break

    return centroids,labels

# Τοποθέτηση των χρωμάτων των κέντρων στην εικόνα
def place_colors(image, centroids, labels):
    new_image = centroids[labels]
    return new_image.reshape(image.shape)

# Αποθήκευση της εικόνας
def save_image(image, path):
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# Κύριος κώδικας
if __name__ == "__main__":
    image_rgb = image_to_rgb("./images-project-1/butterfly.jpg")

    # K-means
    k = 2  # Αριθμός clusters
    centroids, labels = k_means(image_rgb.reshape(-1, 3), k, 50)

    # Τοποθέτηση των χρωμάτων των κέντρων στην εικόνα
    new_image = place_colors(image_rgb, centroids, labels)

    # Αποθήκευση της νέας εικόνας
    save_image(new_image, "output_image.jpg")
