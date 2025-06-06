import torch
from models.cemt_net import CEMTNet
from utils.dataset_loader import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CEMTNet(768, 768, 256).to(device)
model.load_state_dict(torch.load("cemtnet.pth"))
model.eval()

_, val_loader = get_dataloaders()
preds, targets = [], []

with torch.no_grad():
    for x_text, x_audio, labels in val_loader:
        x_text, x_audio = x_text.to(device), x_audio.to(device)
        outputs = model(x_text, x_audio)
        pred = outputs.argmax(dim=1).cpu()
        preds.extend(pred.tolist())
        targets.extend(labels.tolist())

report = classification_report(targets, preds, output_dict=True)
with open("results/cemt_eval_logs/preds.json", "w") as f:
    json.dump(report, f, indent=2)
print("Evaluation done.")
