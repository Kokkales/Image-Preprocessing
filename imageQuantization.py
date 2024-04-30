import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import cv2

def quantImg(image, level):
    # find Xmax and Xmin
    # print(image)
    Xmax = np.max(image)
    Xmin = np.min(image)
    # print(Xmin,Xmax)

    # init new image
    newImageArray = np.zeros_like(image)

    # e.g for level 4 step= (253-0)/(4-1) = 63
    # This means each level would have a range of 64 values. The levels would likely be something like:

    # Level 0: 0-63
    # Level 1: 64-127
    # Level 2: 128-191
    # Level 3: 192-255
    # print(step)
    # for each element of the array
    step = (Xmax - Xmin) / level - 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            currentValue = image[i, j]
            level_index = int((currentValue - Xmin) // step)
            if currentValue < Xmin:
                newImageArray[i, j] = Xmin
            elif currentValue > Xmax:
                newImageArray[i, j] = Xmax
            else:
                newImageArray[i, j] =  Xmin + level_index * step
    # if X[i,j]<Xmin -> r1
    # elif Dk<X[i,j<Dk+1
    # else X[i,j]>Xmax# Calculate the transformation function for the current level
    uniqueArray=np.unique(newImageArray)
    transformation_functions=np.array([])
    for i,item in enumerate(uniqueArray):
        for j in range(int(step)):
            transformation_functions = np.append(transformation_functions,item)
    calculateMSE(imageArray,newImageArray, level)
    return newImageArray, transformation_functions
    # return newImageArray

def calculateMSE(image, newImage, level):
    n = image.shape[0]*image.shape[1]
    MSE = (1 / n) * sum((image[i, j] - newImage[i, j])^2 for i in range(image.shape[0]) for j in range(image.shape[1]))
    print(f"MSE level {level}: {MSE}")

# --------------------------------------------------------PLOT IMAGES
def plotAllImages():
    # Assuming your images are saved in the specified paths
    image_paths = [
        "./images-project-1/barbara.bmp",
        "./results_one/qBarbara_8.bmp",
        "./results_one/qBarbara_12.bmp",
        "./results_one/qBarbara_16.bmp",
        "./results_one/qBarbara_20.bmp",
        "./results_one/qBarbara_40.bmp",
    ]

    # Define the number of rows and columns for the grid layout
    rows = 3
    cols = 2
    # Create a figure and adjust layout spacing
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between subplots

    # Load and display images in the grid with OpenCV
    for i, image_path in enumerate(image_paths):
        row = i // cols  # Integer division for row index
        col = i % cols  # Modulo for column index

        # Load image with OpenCV (assuming BGR format)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{image_path.split('/')[-1]}")  # Extract filename from path
        axes[row, col].axis("off")  # Hide axes for cleaner presentation

    plt.show()
    plt.savefig('final_results.png')

def plotTransformations(caseResults):
    # #-----------------------------------------------------PLOT TRANSFORMATION FUNCTIONS
    Xmax = np.max(imageArray)
    Xmin = np.min(imageArray)

    num_elements = len(caseResults[0][1])
    # Ensure all transformed arrays have the same length
    num_elements_eight = len(caseResults[0][1])
    num_elements_twelve = len(caseResults[1][1])
    num_elements_sixteen = len(caseResults[2][1])
    num_elements_twenty = len(caseResults[3][1])
    # num_elements_fourty = len(caseResults[4][1])
    y_eight = np.linspace(Xmin, Xmax, num_elements_eight)
    y_twelve = np.linspace(Xmin, Xmax, num_elements_twelve)
    y_sixteen = np.linspace(Xmin, Xmax, num_elements_sixteen)
    y_twenty = np.linspace(Xmin, Xmax, num_elements_twenty)
    # y_fourty = np.linspace(Xmin, Xmax, num_elements_fourty)

    # # Create the plot
    plt.plot(caseResults[0][1], y_eight, label="8 Levels")
    plt.plot(caseResults[1][1], y_twelve, label="12 Levels")
    plt.plot(caseResults[2][1], y_sixteen, label="16 Levels")
    plt.plot(caseResults[3][1], y_twenty, label="20 Levels")
    # plt.plot(caseResults[4][1], y_twenty, label="40 Levels")
    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Quantization Levels")

    # Add legend to differentiate lines
    plt.legend()

    # Display the plot
    plt.show()


# import image
try:
    toProcessImage = Image.open("./images-project-1/barbara.bmp")
except OSError:
    raise ValueError("There was a problem with the input image.")

# check if image is grayscale, if not then convert it to grayscale
if toProcessImage.mode!='L':
    toProcessImage=toProcessImage.convert('L')

imageArray = np.array(toProcessImage)
# print(imageArray.shape)

cases=[8,12,16,20, 40]
caseResults=[]
results=[]
for i,case in enumerate(cases):
    caseResults.append(quantImg(imageArray, case))
    results.append((Image.fromarray(caseResults[i][0]),caseResults[i][1]))
    results[i][0].save(f"./results_one/qBarbara_{case}.bmp")
    # print(i,caseResults[i][1])
plotAllImages()
plotTransformations(caseResults)
