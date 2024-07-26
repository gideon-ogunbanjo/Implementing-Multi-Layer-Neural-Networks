# Libraries
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class NN_16_1_Pooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(NN_16_1_Pooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.nn = nn.Sequential(
            nn.Linear(kernel_size * kernel_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x_unfolded = x_unfolded.contiguous().view(b, c, -1, self.kernel_size * self.kernel_size)
        
        out = self.nn(x_unfolded).squeeze(-1)
        out = out.view(b, c, h // self.stride, w // self.stride)
        
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = NN_16_1_Pooling(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 100)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def main():
    start_time = time.time()  # Timer

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    net = Net()
    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(160):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf detected in loss at epoch {epoch}, batch {i}")
                return 
        
        if epoch == 80 or epoch == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training Completed! Time taken: {elapsed_time:.2f} seconds')

if __name__ == '__main__':
    main()
