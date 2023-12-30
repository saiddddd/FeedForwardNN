import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#hyper parameters
input_size = 784 #28x28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

#MNIST
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(),
                                          download = True)

test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

examples = iter(train_loader)
example_data, example_targets = examples.__next__()
print(example_data.shape, example_targets.shape)

for i in range(6):
    
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap = 'gray')
#plt.show()

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

writer.close()
# sys.exit()

class NeuralNet(nn.Module):
      #single hidden layer
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)

    return out #softmax tidak digunakan di layer akhir karena untuk criterion digunakan crossentropyloss (sudah memuat softmax)

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print(model)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

writer.add_graph(model, example_data.reshape(-1, 28*28))
writer.close()
#sys.exit()


#training loop

n_total_step = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    #100, 1, 28, 28
    #100, 784
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)

    #forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)


    #backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    #value, index
    _,predicted = torch.max(outputs, dim = 1)
    running_correct += (predicted == labels).sum().item()

    
    if (i+1) % 100 == 0:
      print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_step}, loss = {loss.item():.4f}')
      writer.add_scalar('training loss', running_loss/100, epoch*n_total_step+i)
      writer.add_scalar('accuracy', running_correct/100, epoch*n_total_step+i)
      
      running_correct = 0
      running_loss = 0.0
      
#test no need the gradient part
labels = []
preds = []

with torch.no_grad():
  n_correct = 0
  n_samples = 0

  for images, labels1 in test_loader:
    images = images.reshape(-1, 28*28).to(device) #karena 1 baris, dan ada 784 = 28*28 features
    labels1 = labels1.to(device)

    outputs = model(images)

    #value, index
    _,predictions = torch.max(outputs, dim = 1)

    n_samples += labels1.size(0)
    n_correct += (predictions == labels1).sum().item()
    
    class_predictions = [F.softmax(output, dim  = 0) for output in outputs]
    
    preds.append(class_predictions)
    labels.append(predictions)
   
  preds = torch.cat([torch.stack(batch) for batch in preds])  
  labels = torch.cat(labels)

  acc = 100.0 * n_correct/n_samples
  print(f'accuracy on the test images = {acc} %')
  
  classes = range(10)
  for i in classes:
      labels_i = labels == i
      preds_i = preds[:,i]
      writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
      writer.close()
      
      
      
      
      
      
  

