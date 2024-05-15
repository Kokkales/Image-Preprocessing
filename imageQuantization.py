import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# A
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
            levelIndex = int((currentValue - Xmin) // step)
            if currentValue < Xmin:
                newImageArray[i, j] = Xmin
            elif currentValue > Xmax:
                newImageArray[i, j] = Xmax
            else:
                newImageArray[i, j] =  Xmin + levelIndex * step
    # if X[i,j]<Xmin -> r1
    # elif Dk<X[i,j<Dk+1
    # else X[i,j]>Xmax# Calculate the transformation function for the current level
    uniqueArray=np.unique(newImageArray)
    transformationFunctions =np.array([])
    for i,item in enumerate(uniqueArray):
        for j in range(int(step)):
            transformationFunctions  = np.append(transformationFunctions ,item)
    calculateMSE(imageArray,newImageArray, level)
    return newImageArray, transformationFunctions
    # return newImageArray

# Β
def plotTransformations(caseResults):
    # #-----------------------------------------------------PLOT TRANSFORMATION FUNCTIONS
    Xmax = np.max(imageArray)
    Xmin = np.min(imageArray)

    numElements = len(caseResults[0][1])
    # Ensure all transformed arrays have the same length
    numElementsEight = len(caseResults[0][1])
    numElementsTwelve = len(caseResults[1][1])
    numElementsSixteen = len(caseResults[2][1])
    numElementsTwenty = len(caseResults[3][1])
    # numElementsfourty = len(caseResults[4][1])
    yEight = np.linspace(Xmin, Xmax, numElementsEight)
    yTwelve = np.linspace(Xmin, Xmax, numElementsTwelve)
    ySixteen = np.linspace(Xmin, Xmax, numElementsSixteen)
    yTwenty = np.linspace(Xmin, Xmax, numElementsTwenty)
    # y_fourty = np.linspace(Xmin, Xmax, numElementsFourty)

    # # Create the plot
    plt.plot(caseResults[0][1], yEight, label="8 Levels")
    plt.plot(caseResults[1][1], yTwelve, label="12 Levels")
    plt.plot(caseResults[2][1], ySixteen, label="16 Levels")
    plt.plot(caseResults[3][1], yTwenty, label="20 Levels")
    # plt.plot(caseResults[4][1], y_twenty, label="40 Levels")
    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Quantization Levels")
    plt.legend()
    plt.savefig('./results_one/transformations_plot.png')
    # Display the plot
    plt.show()

# Γ
def plotAllImages():
    imagePaths = [
        "./images-project-1/barbara.bmp",
        "./results_one/qBarbara_8.bmp",
        "./results_one/qBarbara_12.bmp",
        "./results_one/qBarbara_16.bmp",
        "./results_one/qBarbara_20.bmp",
        "./results_one/qBarbara_40.bmp",
    ]

    rows = 3
    cols = 2

    # Create a figure and adjust layout spacing
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between subplots

    for i, imagePath in enumerate(imagePaths):
        row = i // cols  # Integer division for row index
        col = i % cols  # Modulo for column index

        # Load image with OpenCV (assuming BGR format)
        image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{imagePath.split('/')[-1]}")  # Extract filename from path
        axes[row, col].axis("off")  # Hide axes for cleaner presentation

    plt.savefig('./results_one/final_results.png')
    plt.show()

# Δ
def calculateMSE(image, newImage, level):
    MSE = np.mean((image.astype(np.float32) - newImage.astype(np.float32)) ** 2)
    print(f"MSE level {level}: {MSE}")


if __name__=='__main__':
    # import image
    imageArray = cv2.imread('./images-project-1/barbara.bmp', cv2.IMREAD_GRAYSCALE)
    cases=[8,12,16,20,40]
    caseResults=[]
    results=[]
    for i,case in enumerate(cases):
        caseResults.append(quantImg(imageArray, case))
        results.append((Image.fromarray(caseResults[i][0]),caseResults[i][1]))
        results[i][0].save(f"./results_one/qBarbara_{case}.bmp")
        # print(i,caseResults[i][1])
    plotTransformations(caseResults)
    plotAllImages()
