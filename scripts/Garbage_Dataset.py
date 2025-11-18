from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from pprint import pprint
from PIL import Image
import random
import torch
import json
import cv2
import os

class GarbageDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        split_dir = os.path.join(root, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.coco = coco
        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = sorted(coco["categories"], key=lambda c: c["id"])
        self.id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        self.num_classes = len(self.categories) + 1

        # map: image_id -> list annotation
        self.img_id_to_anns = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            self.img_id_to_anns[ann["image_id"]].append(ann)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.root, self.split, file_name)
        img = Image.open(img_path).convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: [x_min, y_min, width, height] (pixel)
            boxes.append([x, y, x + w, y + h])  # chuyển sang [x1, y1, x2, y2]
            labels.append(ann["category_id"])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # lật ảnh
            image = F.hflip(image)
            w = image.shape[-1]  # sau ToTensor: [C, H, W]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                x1 = boxes[:, 0]
                x2 = boxes[:, 2]
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
                target["boxes"] = boxes
        return image, target

def get_transform(train: bool = True):
    transforms = [ToTensor()]
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
    
if __name__ == "__main__":
    dataset = GarbageDataset(root="data/Garbage", split="train", transforms=None)
    img, target = dataset[0]
    img.show()
    pprint(target)