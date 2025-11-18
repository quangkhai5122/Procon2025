from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

def detection_collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def calculate_map(model, data_loader, device, coco_gt, show_progress=False, desc="Evaluating"):
    """Calculate mAP using COCO evaluation"""
    model.eval()
    results = []

    iterator = tqdm(data_loader, desc=desc, leave=False) if show_progress else data_loader
    try:
        with torch.no_grad():
            for images, targets in iterator:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for target, output in zip(targets, outputs):
                    image_id = target["image_id"].item()
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        results.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score)
                        })
    finally:
        if show_progress and hasattr(iterator, "close"):
            iterator.close()
    
    if len(results) == 0:
        return 0.0
    
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]  # mAP @ IoU=0.50:0.95

def calculate_confusion_matrix(
    model,
    data_loader,
    device,
    num_classes,
    iou_threshold=0.5,
    score_threshold=0.5,
    show_progress=False,
    desc="Calculating CM",
):
    """Calculate confusion matrix for object detection"""
    model.eval()

    # Initialize confusion matrix (num_classes-1 because we exclude background)
    cm = np.zeros((num_classes - 1, num_classes - 1), dtype=np.int32)

    iterator = tqdm(data_loader, desc=desc, leave=False) if show_progress else data_loader
    try:
        with torch.no_grad():
            for images, targets in iterator:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for target, output in zip(targets, outputs):
                    gt_boxes = target["boxes"].cpu().numpy()
                    gt_labels = target["labels"].cpu().numpy()

                    pred_boxes = output["boxes"].cpu().numpy()
                    pred_scores = output["scores"].cpu().numpy()
                    pred_labels = output["labels"].cpu().numpy()

                    # Filter predictions by score threshold
                    mask = pred_scores >= score_threshold
                    pred_boxes = pred_boxes[mask]
                    pred_labels = pred_labels[mask]

                    # Match predictions to ground truth
                    matched_gt = set()

                    for pred_box, pred_label in zip(pred_boxes, pred_labels):
                        best_iou = 0
                        best_gt_idx = -1

                        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                            if gt_idx in matched_gt:
                                continue

                            # Calculate IoU
                            x1 = max(pred_box[0], gt_box[0])
                            y1 = max(pred_box[1], gt_box[1])
                            x2 = min(pred_box[2], gt_box[2])
                            y2 = min(pred_box[3], gt_box[3])

                            if x2 > x1 and y2 > y1:
                                intersection = (x2 - x1) * (y2 - y1)
                                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                                union = pred_area + gt_area - intersection
                                iou = intersection / union

                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = gt_idx

                        # If IoU > threshold, match prediction to ground truth
                        if best_iou >= iou_threshold and best_gt_idx != -1:
                            matched_gt.add(best_gt_idx)
                            gt_label = gt_labels[best_gt_idx]
                            cm[gt_label - 1, pred_label - 1] += 1
                        # else: False positive (ignored for CM simplicity)

                    # Unmatched ground truths are false negatives (ignored in CM aggregation)
    finally:
        if show_progress and hasattr(iterator, "close"):
            iterator.close()

    return cm