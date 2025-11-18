from torchvision.transforms import ToTensor, Compose, Resize, RandomAffine, ColorJitter, RandomHorizontalFlip
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import detection_collate_fn, plot_confusion_matrix, calculate_map, calculate_confusion_matrix
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import torch
import os
from Garbage_Dataset import (
    GarbageDataset,
    get_transform,
)

if __name__ == "__main__":
    # Config
    data_root = "./data/Garbage"  
    num_classes = 6  # 5 lớp rác + 1 background
    num_epochs = 100
    batch_size = 4
    lr = 5e-3
    momentum = 0.9
    weight_decay = 0.0005
    checkpoint_dir = "./checkpoints"
    log_dir = "./runs/fasterrcnn"
    resume_from = "./checkpoints/last_model_fasterrcnn.pth"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    train_dataset = GarbageDataset(
        root=data_root,
        split="train",
        transforms=get_transform(train=True),
    )
    val_dataset = GarbageDataset(
        root=data_root,
        split="valid",
        transforms=get_transform(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,         
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate_fn,
    )
    
    # COCO GT for mAP calculation
    train_ann_path = os.path.join(data_root, "train", "_annotations.coco.json")
    val_ann_path = os.path.join(data_root, "valid", "_annotations.coco.json")
    coco_train = COCO(train_ann_path)
    coco_val = COCO(val_ann_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Adaptive weight decay: giảm weight decay theo epoch
    initial_weight_decay = weight_decay
    
    best_map = 0.0
    best_epoch = 0
    start_epoch = 0
    
    # Resume training
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f"Resuming from epoch {start_epoch}, best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        current_weight_decay = initial_weight_decay * (0.95 ** epoch)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_weight_decay
        
        model.train()
        total_loss = 0.0
        
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix({"loss": f"{losses.item():.4f}"})
        
        pbar.close()  

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = total_loss / max(1, len(train_loader))

        # Calculate train mAP (suppresses tqdm in helper)
        train_map = calculate_map(model, train_loader, device, coco_train)
        model.train()  # Restore training mode for loss-only validation
        
        # Validation loop
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, targets in pbar_val:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                pbar_val.set_postfix({"loss": f"{losses.item():.4f}"})
        pbar_val.close()
        val_loss /= max(1, len(val_loader))

        # Calculate val mAP
        val_map = calculate_map(model, val_loader, device, coco_val)
        
        # Calculate confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0:
            cm = calculate_confusion_matrix(model, val_loader, device, num_classes)
            class_names = [train_dataset.id_to_name[i+1] for i in range(num_classes - 1)]
            plot_confusion_matrix(writer, cm, class_names, epoch)

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mAP/train', train_map, epoch)
        writer.add_scalar('mAP/val', val_map, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Weight_Decay', current_weight_decay, epoch)

        print(f"[Epoch {epoch + 1}/{num_epochs}]")
        print(f"  Train - Loss: {avg_loss:.4f}, mAP: {train_map:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
        print(f"  LR: {current_lr:.6f}, WD: {current_weight_decay:.6f}")
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'best_map': best_map,
            }, os.path.join(checkpoint_dir, "best_model_fasterrcnn.pth"))
            print(f"  Best model saved! mAP: {best_map:.4f}")
        
        # Save last model (checkpoint) after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_map': best_map,
        }, os.path.join(checkpoint_dir, "last_model_fasterrcnn.pth"))
        print()  # Thêm dòng trống giữa các epoch
    
    print(f"\nTraining completed!")
    print(f"Best mAP: {best_map:.4f} at epoch {best_epoch}")
    
    writer.close()
