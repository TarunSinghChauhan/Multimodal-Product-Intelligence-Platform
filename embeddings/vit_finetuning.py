import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTConfig
import mlflow

# Configuration
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 0.01

class ProductDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.categories = sorted(df['category'].unique().tolist())
        self.cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGE_DIR, row['image_filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.cat_to_idx[row['category']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_vit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Data
    df_products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    df_mapping = pd.read_csv(os.path.join(DATA_DIR, "product_image_mapping.csv"))
    df = df_products.merge(df_mapping, on='product_id')
    
    categories = sorted(df['category'].unique().tolist())
    num_labels = len(categories)

    # Split Data (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['category'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=42)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ProductDataset(train_df, transform=train_transform)
    val_ds = ProductDataset(val_df, transform=val_transform)
    test_ds = ProductDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # MLflow Setup
    mlflow.set_experiment("ViT_Product_Classification")

    best_val_acc = 0
    patience = 3
    no_improve = 0

    with mlflow.start_run():
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        })

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix(loss=train_loss/len(train_loader), acc=100.*correct/total)

            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100.*val_correct/val_total
            print(f"Val Acc: {val_acc:.2f}%")
            
            mlflow.log_metrics({
                "train_loss": train_loss/len(train_loader),
                "train_acc": 100.*correct/total,
                "val_loss": val_loss/len(val_loader),
                "val_acc": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_of_dict(), os.path.join(MODELS_DIR, "best_vit_model.pth"))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered")
                    break

        # Final Evaluation on Test Set
        print("Evaluating on test set...")
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_vit_model.pth")))
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images).logits
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Metrics
        report = classification_report(all_labels, all_preds, target_names=categories)
        print(report)
        with open(os.path.join(MODELS_DIR, "classification_report.txt"), "w") as f:
            f.write(report)
            
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
        
        mlflow.log_artifact(os.path.join(MODELS_DIR, "confusion_matrix.png"))
        mlflow.log_artifact(os.path.join(MODELS_DIR, "classification_report.txt"))

if __name__ == "__main__":
    train_vit()
