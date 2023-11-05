#Srinivs Peri & Sriram Pandi

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

#Hyper Parameteres
n_epochs = 5
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 42
   
torch.backends.cudnn.enabled = False   # GPU Off
torch.manual_seed(random_seed) 

# Loading Data
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('C:/Users/raghu', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('C:/Users/raghu', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=batch_size_test, shuffle=True)

#Building Network model     
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(320, 50)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(50, 10)
    

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        x = self.pool2(self.relu(x))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


 
network = MyNet()
summary(network, input_size=(1, 28, 28))

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


   

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]  

#Train data models
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]/tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')

#-------------------------------------------------------------------------------------------------------------
#Test data Model
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))



def main():
   
   print(example_data.shape) # Prints the sahpe of the data set.
   
   
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
   
   with torch.no_grad():
      output = network(example_data)
   
   fig = plt.figure()
   for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
         output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
   fig
   plt.show()

   return

if __name__ == "__main__":
    main()
    

