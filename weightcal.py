#Srinivs Peri Sriram Pandi

import torch
import torchvision
import matplotlib.pyplot as plt
from pytorchexample import MyNet
import cv2
import matplotlib.pyplot as plt


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('C:/Users/raghu', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=50, shuffle=True)
# Load the model
model = MyNet()
model.load_state_dict(torch.load('results/model.pth'))

# Get the weights of the first layer
weights = model.conv1.weight
filters = model.conv1.weight.detach().numpy()

# Print the filter weights and their shape
print("Filter weights:", weights)
print("Filter shape:", weights.shape)


dataiter = iter(test_loader)
images, labels = next(dataiter)

# Create a 3x4 grid of plots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 8))
fig.suptitle('Filters in Convolutional Layer 1')

# Plot each filter
for i, ax in enumerate(axes.flat):
    if i < 10:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(filters[i][0])
        # Remove the x and y ticks
      
plt.tight_layout()  
# Display the plot
plt.show()

with torch.no_grad():
    weights = model.conv1.weight

# apply the 10 filters to the first training example image using OpenCV's filter2D function
with torch.no_grad():
    example = next(iter(test_loader))[0]  # get the first training example
    example_np = example.numpy()[0, 0]  # convert it to a numpy array and extract the grayscale image
    filtered_images = []
    for i in range(10):
        filter_np = weights[i, 0].numpy()  # extract the ith filter and convert it to a numpy array
        filtered_image_np = cv2.filter2D(example_np, -1, filter_np)  # apply the filter using OpenCV's filter2D function
        filtered_images.append(filtered_image_np)

# plot the 10 filtered images
fig, axes = plt.subplots(nrows=5, ncols=2)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(filtered_images[i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

