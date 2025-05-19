import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 10)

        self._fc1_in_features = 32 * 14 * 14

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        if x.size(1) != self._fc1_in_features:
            self._fc1_in_features = x.size(1)
            self.fc1 = nn.Linear(self._fc1_in_features, 100).to(x.device)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def train_model(model, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model

def prune_smallcnn(model, prune_ratio):
    model = copy.deepcopy(model)
    model.to(device)

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(dummy_input)
    loss = output.mean()
    loss.backward()

    scores = model.conv2.weight * model.conv2.weight.grad
    importance = scores.abs().mean(dim=(1, 2, 3))
    num_keep = max(1, int((1 - prune_ratio) * importance.numel()))
    keep_idx = importance.topk(num_keep).indices

    model.conv2.out_channels = len(keep_idx)
    model.conv2.weight = nn.Parameter(model.conv2.weight.data[keep_idx])
    if model.conv2.bias is not None:
        model.conv2.bias = nn.Parameter(model.conv2.bias.data[keep_idx])

    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        dummy_output = model.pool(F.relu(model.conv2(F.relu(model.conv1(dummy_input)))))
        new_in_features = dummy_output.view(1, -1).size(1)

    model.fc1 = nn.Linear(new_in_features, 100)
    model._fc1_in_features = new_in_features

    return model

def run_experiment(prune_ratio=None):
    model = SmallCNN()
    model = train_model(model, epochs=10)

    if prune_ratio:
        model = prune_smallcnn(model, prune_ratio)
        model = train_model(model, epochs=3)

    acc = evaluate(model)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.to('cpu'), (1, 28, 28), print_per_layer_stat=False, as_strings=False)
    return acc, params, macs

for label, ratio in zip(["Full Model", "50% Pruned", "70% Pruned"], [0.0, 0.5, 0.7]):
    acc, params, flops = run_experiment(prune_ratio=ratio if ratio > 0 else None)
    print(f"[{label}] Accuracy: {acc:.2f}%, Params: {params / 1e6:.4f}M, FLOPs: {flops / 1e6:.2f}M")
