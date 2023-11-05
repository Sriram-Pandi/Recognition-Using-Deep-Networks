#Srinivs Peri & Sriram Pandi

import torch
import torchvision
import matplotlib.pyplot as plt
from pytorchexample import MyNet
 
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('C:/Users/raghu', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=10, shuffle=True)

dataiter = iter(test_loader)
images, labels = next(dataiter)

modeltest = MyNet()
modeltest.load_state_dict(torch.load('results/model.pth'))
modeltest.eval()

# Run the examples through the network and print the predictions
with torch.no_grad():
    outputs = modeltest(images)
    _, predicted = torch.max(outputs, 1)
    for i in range(10):
        print("Example", i+1)
        print("Output values:", ", ".join([f"{value:.2f}" for value in torch.exp(outputs[i])]))
        print("Prediction:", predicted[i].item())
        print("Label:", labels[i].item())

# Plot the first 9 digits with the predictions below each image
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,8))
for i, ax in enumerate(axes.flat):
    if i < 9:
        ax.imshow(images[i][0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {predicted[i].item()}\nLabel: {labels[i].item()}")
plt.tight_layout()
plt.show()