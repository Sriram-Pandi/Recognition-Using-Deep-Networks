#Srinivs Peri & Sriram Pandi
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# Load pre-trained ResNet-18 network
resnet = models.resnet18(pretrained=True)

# Define a new model that only includes the first 10 convolutional layers of ResNet-18
new_model = nn.Sequential(*list(resnet.children())[:10])

# Print the new model
print(new_model)

# Evaluate the first couple of convolutional layers of the new model
for layer in new_model:
    if isinstance(layer, nn.Conv2d):
        print(layer)
        break


# Get the weights of the first convolutional layer
weights = resnet.conv1.weight

# Print the shape of the weights
print(weights.shape)
filter_names = [f'Filter {i+1}' for i in range(weights.shape[0])]

# Visualize the first 10 filters
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axs.flat):
    if i < 10:
        ax.imshow(weights[i].detach().numpy().transpose(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(filter_names[i])
plt.show()