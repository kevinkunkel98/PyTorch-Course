import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# select gpu as divice

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {device}")

# data loading and transformation
# CIFAR-10: 3 Kanäle (RGB) → Normalisierung pro Kanal

# Val/Test: nur normalisieren
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Training: Data Augmentation → mehr Varianz → bessere Generalisierung
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),             # 50% horizontal spiegeln
    transforms.RandomCrop(32, padding=4),           # zufällig zuschneiden mit 4px Rand
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Helligkeit/Kontrast variieren
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
# Val braucht eigenen Datensatz OHNE Augmentation
val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Train/Val Split: 45000 Training, 5000 Validation
train_data, val_data_unused = random_split(train_dataset, [45000, 5000])
_, val_data = random_split(val_dataset, [45000, 5000])

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# CIFAR-10 Klassen
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # ===== Block 1: Einfache Kanten & Farben erkennen =====
        # Conv → BatchNorm → ReLU → MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # 3×32×32 → 32×32×32
        self.bn1 = nn.BatchNorm2d(32)  # Normalisiert die Ausgabe → stabileres Training

        # ===== Block 2: Komplexere Muster (Ecken, Texturen) =====
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32×16×16 → 64×16×16
        self.bn2 = nn.BatchNorm2d(64)

        # ===== Block 3: Noch komplexere Features (Formen, Teile) =====
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64×8×8 → 128×8×8
        self.bn3 = nn.BatchNorm2d(128)

        # ===== Fully Connected =====
        # Nach 3× MaxPool: 32→16→8→4, also 128*4*4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)  # 50% Dropout — stärker weil größeres Netz

    def forward(self, x):
        # Block 1: (batch, 3, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))    # → (batch, 32, 32, 32)
        x = F.max_pool2d(x, 2)                 # → (batch, 32, 16, 16)

        # Block 2:
        x = F.relu(self.bn2(self.conv2(x)))    # → (batch, 64, 16, 16)
        x = F.max_pool2d(x, 2)                 # → (batch, 64, 8, 8)

        # Block 3:
        x = F.relu(self.bn3(self.conv3(x)))    # → (batch, 128, 8, 8)
        x = F.max_pool2d(x, 2)                 # → (batch, 128, 4, 4)

        # Flatten + FC
        x = x.view(-1, 128 * 4 * 4)           # → (batch, 2048)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))                # → (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)                        # → (batch, 10)
        return x


model = SimpleCNN().to(device)
print(f"Model summary: {model}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Scheduler: sanfter — alle 5 Epochen × 0.5 (statt ×0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)
    return avg_loss, accuracy

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Scheduler: LR nach jeder Epoch anpassen
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    avg_loss = running_loss / len(train_loader)
    _, train_acc = evaluate_model(model, train_loader, criterion)
    _, val_acc = evaluate_model(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, LR: {current_lr}")

# Finale Evaluation auf Test-Daten (nur einmal am Ende!)
_, test_acc = evaluate_model(model, test_loader, criterion)
print(f"\n✅ Training fertig! Test Acc: {test_acc:.4f}")
