import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap
from torch.utils.data import TensorDataset, DataLoader,random_split, Subset
from torchvision import datasets
from torchvision.transforms import transforms


batch_size = 256

kwargs = {'num_workers': 1}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn6   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn7   = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.bn8   = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn9   = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn10  = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn11   = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn12   = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.bn13   = nn.BatchNorm2d(512)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2 * 2 * 512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 100)

    def forward(self, x):
        # Layer 1 64
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # Layer 2 128
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # Layer 3 256
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # Layer 5 512
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # Layer 6 512
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))

        # Layer 7 Fully Connected Layer
        x = x.view(x.size(0),-1)  # flatten / reshape
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


train_data_for_stats = datasets.CIFAR100(root='data', train=True, download=True, transform=transforms.ToTensor())
stats_loader = DataLoader(train_data_for_stats, batch_size=batch_size, shuffle=False)


# Compute mean and std
n_samples_seen = 0
mean = torch.zeros(3)
std = torch.zeros(3)
for train_batch, train_target in stats_loader:
    batch_size = train_batch.shape[0]
    train_batch = train_batch.view(batch_size, 3, -1)
    this_mean = train_batch.mean(dim=2)
    this_std  = train_batch.std(dim=2)
    mean += torch.sum(this_mean, dim=0)
    std += torch.sum(this_std, dim=0)
    n_samples_seen += batch_size

mean /= n_samples_seen
std /= n_samples_seen


# Define transforms using computed mean and std
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])

# Load datasets another time with normalization transforms
train_data_aug = datasets.CIFAR100(
    root='data', train=True, download=True,
    transform=transform_train
)

val_data_no_aug = datasets.CIFAR100(
    root='data', train=True, download=True,
    transform=transform_test
)

test_data = datasets.CIFAR100(
    root='data', train=False, download=True,
    transform=transform_test
)

total_size = len(train_data_aug)

val_split_size = total_size // 10 # 50,000 -> 5000
train_split_size = total_size - val_split_size # 45,000

generator = torch.Generator().manual_seed(0) # For reproducibility

# Split training data into validation and training sets
perm = torch.randperm(total_size, generator=generator).tolist()

train_idx = perm[:train_split_size]
val_idx   = perm[train_split_size:]

train_subset = Subset(train_data_aug, train_idx)        # AUGMENTED
validation_subset = Subset(val_data_no_aug, val_idx)    # NO AUG

# Create loaders for training and testing
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **kwargs)
validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, **kwargs)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

def train(model, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # sum up batch loss
            _, pred = output.data.max(dim=1)
            # get the index of the max log-probability
            correct += torch.sum(pred == target.data.long()).item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = float(correct) / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f},'
              ' Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_accuracy))
    return test_loss, test_accuracy


def evaluate(model, loader):
    model.eval()
    eval_loss  = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            eval_loss += F.nll_loss(output, target, reduction="sum").item()
            _, pred = output.data.max(dim=1)
            correct += torch.sum(pred == target.data.long()).item()

        eval_loss /= len(loader.dataset)
        test_accuracy = float(correct) / len(loader.dataset)


    return eval_loss, test_accuracy

log_interval = 100
epochs = 100
lr = 0.1 # changed to 0.1 from 0.01

model = Model().to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=lr,  momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[28, 40, 50, 75, 90],
    gamma=0.2
)


logs = {'epoch': [], 'train_loss': [], 'test_loss': [],
        'test_accuracy': [], 'lr': []}

lr_drops = 0 #  VGG-16 style
prev_lr = optimizer.param_groups[0]["lr"] #  VGG-16 style


for epoch in range(epochs):
    train_loss = train(model, optimizer, train_loader, epoch)
    val_loss, val_accuracy = evaluate(model, validation_loader)
    test_loss, test_accuracy = test(model, test_loader)
    logs['epoch'].append(epoch)
    logs['train_loss'].append(train_loss)
    logs['test_loss'].append(test_loss)
    logs['test_accuracy'].append(test_accuracy)
    logs['lr'].append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy*100:.2f}% | lr={optimizer.param_groups[0]['lr']:.6f}")

    #scheduler.step(val_accuracy) # VGG-16 style.
    scheduler.step()


    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr < prev_lr:
        lr_drops += 1
        prev_lr = new_lr

    print(f"LR drop #{lr_drops}: now {new_lr:g}")

    if lr_drops >= 6:
        print("Stopping after 6 LR drops .")
        break

