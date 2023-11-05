#Srinivs Peri & Sriram Pandi
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pytorchexample import MyNet
import torch.optim as optim


#Hyper Parameteres
n_epochs = 9
learning_rate = 0.01
momentum = 0.5
log_interval = 3
random_seed = 42

torch.backends.cudnn.enabled = False   # GPU Off
torch.manual_seed(random_seed) 

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


# create a dataloader to iterate over the dataset
greek_train = torch.utils.data.DataLoader( torchvision.datasets.ImageFolder('C:/visualstudpr/greek_train', 
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 6,
        shuffle = True )
greek_test = torch.utils.data.DataLoader( torchvision.datasets.ImageFolder("C:/visualstudpr/greeks", 
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 10,
        shuffle = True )

examples = enumerate(greek_test)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape) 

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

#Load Model
model = MyNet()
model.load_state_dict(torch.load('results/model.pth'))

# Freeze the network weights
for param in model.parameters():
   param.requires_grad = False

model.fc2 = nn.Linear(50, 3)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Print the modified model
print(model)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(greek_train.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(greek_train):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(greek_train.dataset),
        100. * batch_idx / len(greek_train), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*6) + ((epoch-1)*len(greek_train.dataset)))


def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in greek_test:
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(greek_test.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(greek_test.dataset),
    100. * correct / len(greek_test.dataset)))
  

def main():
   
   test()
   for epoch in range(1, n_epochs + 1):
      train(epoch)
      test()

   fig = plt.figure()
   plt.plot(train_counter, train_losses, color='blue')
   plt.scatter(test_counter, test_losses, color='red')
   plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
   plt.xlabel('number of training examples seen')
   plt.ylabel('negative log likelihood loss')
   fig
   

   fig = plt.figure()
   with torch.no_grad():
      output = model(example_data)
      
      for i in range(6):
         plt.subplot(2,3,i+1)
         plt.tight_layout()
         plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
         predicted_class = output.data.max(1, keepdim=True)[1][i].item()
         if predicted_class == 0:
            plt.title("Prediction: alpha")
         elif predicted_class == 1:
            plt.title("Prediction: beta")
         elif predicted_class == 2:
            plt.title("Prediction: gama")
            
         plt.xticks([])
         plt.yticks([])
      plt.show()

  
   return


if __name__ == "__main__":
   main()
   



