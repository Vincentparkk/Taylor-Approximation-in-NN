import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

class PrunedLeNet5(nn.Module):
    def __init__(self, conv1_out=6, conv2_out=16, fc1_out=120, fc2_out=84):
        super(PrunedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, 5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 5)
        self.fc1 = nn.Linear(conv2_out * 4 * 4, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_data_loaders():
    transform = transforms.ToTensor()
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform),
                             batch_size=1000)
    return train_loader, test_loader

def train(model, loader, optimizer, criterion, device, epochs=3):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
    return correct / len(loader.dataset)

def taylor_prune(model, ratio):
    pruned_cfg = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)) and layer.weight.grad is not None:
            score = torch.abs(layer.weight.data * layer.weight.grad)
            score = score.view(score.size(0), -1).mean(dim=1)
            keep = torch.topk(score, int(score.size(0) * (1 - ratio)), largest=True).indices
            pruned_cfg[name] = keep
    return pruned_cfg

def rebuild_model(cfg):
    return PrunedLeNet5(
        conv1_out=len(cfg['conv1']),
        conv2_out=len(cfg['conv2']),
        fc1_out=120,
        fc2_out=84,
    )

def transfer_conv_weights(old, new, cfg):
    device = next(old.parameters()).device
    new.conv1.weight.data = old.conv1.weight.data[cfg['conv1']].clone()
    new.conv1.bias.data = old.conv1.bias.data[cfg['conv1']].clone()
    new.conv2.weight.data = old.conv2.weight.data[cfg['conv2']][:, cfg['conv1']].clone()
    new.conv2.bias.data = old.conv2.bias.data[cfg['conv2']].clone()

def get_model_info(model):
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=False, print_per_layer_stat=False, verbose=False)
    return params / 1e6, macs / 1e6

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()

    configs = {
        "original": 0.0,
        "pruned_50%": 0.5,
        "pruned_70%": 0.7
    }

    print(f"{'Model':<15} {'Accuracy':<10} {'Params':<10} {'FLOPs':<10}")
    print("-" * 50)

    for name, ratio in configs.items():
        model = PrunedLeNet5().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, train_loader, optimizer, criterion, device)

        if name == "original":
            acc = evaluate(model, test_loader, device)
            p, f = get_model_info(model)
            print(f"{name:<15} {acc:.4f}     {p:.4f}M    {f:.4f}M")
        else:
            model.eval()
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            model.zero_grad()
            loss.backward()

            cfg = taylor_prune(model, ratio)
            cfg = {k: cfg[k].to(device) for k in ['conv1', 'conv2']}
            new_model = rebuild_model(cfg).to(device)
            transfer_conv_weights(model, new_model, cfg)

            optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)
            train(new_model, train_loader, optimizer, criterion, device, epochs=2)

            acc = evaluate(new_model, test_loader, device)
            p, f = get_model_info(new_model)
            print(f"{name:<15} {acc:.4f}     {p:.4f}M    {f:.4f}M")

run_experiment()
