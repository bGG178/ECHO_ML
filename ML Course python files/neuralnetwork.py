import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.testing.jpl_units import Epoch

#Device GPU config to run tensors on GPU instead of CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
input_size = 784 # input image 28x28
hidden_size = 1010
num_classes = 10 #0-9
num_epochs = 3
batch_size = 300
learning_rate = 0.002

#MNIST dataset

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

#Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

examples = iter(test_loader)

example_data, example_targets = next(examples)

for i in range(6):

    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

#Fully connected NN
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out=self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#Training
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Epoch {epoch}, {i+1}/{n_total_steps}, L: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        img_path = "911.png"
        image = Image.open(img_path).convert('L')
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),  # Converts to [0,1] and shape [1,28,28]
            #transforms.Normalize((0.1307,), (0.3081,))  # Normalize like MNIST, if you trained that way
        ])
        img_tensor = transform(image).to(device)  # shape: [1, 28, 28]
        img_tensor = img_tensor.view(-1, 28 * 28)  # flatten to [1, 784]


        chrisoutput = model(img_tensor)




        _, predicted = torch.max(outputs,1)

        _,chrispredicted = torch.max(chrisoutput,1)

        if n_correct == 0:


            image = img_tensor.cpu().reshape(28, 28)  # convert back to 28x28
            plt.imshow(image, cmap='gray')
            print(chrispredicted)
            plt.title(f'Predicted: {chrispredicted[0]}, Actual: 9/11')
            plt.axis('off')
            plt.show()
        n_correct += (predicted == labels).sum().item()

    acc = n_correct/n_samples
    print(f'Final accuracy on {n_samples} is {100*acc}%')