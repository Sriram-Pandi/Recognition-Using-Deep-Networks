#Srinivs Peri & Sriram Pandi

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pytorchexample import MyNet
import torch.optim as optim
import os
from PIL import Image, ImageOps

import numpy as np
 
torch.backends.cudnn.enabled = False   # GPU Off
num_images_plotted = 0
model = MyNet()
model.load_state_dict(torch.load('results/model.pth'))
model.eval()

# Define the path to the folder containing the images
folder_path = 'C:/visualstudpr/Handwrittendataset/cropped_1'

# Loop through all the images in the folder
for filename in os.listdir(folder_path):
    # Open the image and convert it to grayscale
    with Image.open(os.path.join(folder_path, filename)).convert('L') as img:
        # Resize the image to 28x28 if necessary
        if img.size != (28, 28):
            img = img.resize((28, 28))
        
        # Invert the intensities if necessary to match the MNIST dataset
        img = ImageOps.invert(img)
        
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Convert the numpy array to a torch tensor
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)
        
        # Normalize the tensor to match the MNIST dataset
        img_tensor = img_tensor / 255.0
        
        # Run the tensor through the model and print the prediction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.data.max(1, keepdim=True)[1][0].item()
            print(f"Prediction for {filename}: {prediction}")

                    # Plot the image and prediction
        if num_images_plotted < 9:
            plt.subplot(3, 3, num_images_plotted+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Prediction: {prediction}")
            plt.axis('off')
            num_images_plotted += 1
        else:
            break

# Show the plot
plt.tight_layout()
plt.show()
