import torch
import torch.nn as nn
import torch.optim as optim
from models.cemt_net import CEMTNet
from utils.dataset_loader import get_dataloaders
from sklearn.metrics import accuracy_score, f1_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
text_dim, audio_dim, hidden_dim = 768, 768, 256
batch_size = 16
data_path = "/root/autodl-tmp/data/preprocess/DAIC-WOZ"

# 模型构建
model = CEMTNet(text_dim, audio_dim, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 加载数据
train_loader, val_loader = get_dataloaders(data_path, batch_size=batch_size)

# 训练过程
best_val_f1 = 0
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for x_text, x_audio, labels in train_loader:
        x_text, x_audio, labels = x_text.to(device), x_audio.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(x_text, x_audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证阶段
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_text, x_audio, labels in val_loader:
            x_text, x_audio = x_text.to(device), x_audio.to(device)
            outputs = model(x_text, x_audio)
            pred = outputs.argmax(dim=1).cpu()
            preds.extend(pred.tolist())
            targets.extend(labels.tolist())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')
    print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

    # 保存模型
    if f1 > best_val_f1:
        best_val_f1 = f1
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/cemtnet.pth")
        print(f"📌 Best model updated at epoch {epoch} (F1: {f1:.4f})")
