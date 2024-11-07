import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric, ConfusionMatrix

def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epochs: int = 30,
        lr: float = 5e-3,
        batch_size: int = 256,
        seed: int = 2024,
        **kwargs
):
    
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # load data  
    train_loader = load_data("road_data/train", transform_pipeline="default", return_dataloader=True, num_workers=0, batch_size=batch_size, shuffle=True)
    val_loader = load_data("road_data/val", return_dataloader=True, num_workers=0, batch_size=batch_size, shuffle=False)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name)
    model = model.to(device)

    # Define loss functions
    criterion_segmentation = torch.nn.CrossEntropyLoss()
    criterion_depth = torch.nn.MSELoss()

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0

    # Initialize DetectionMetric and ConfusionMatrix to compute metrics
    detection_metrics = DetectionMetric(num_classes=3)
    confusion_matrix = ConfusionMatrix(num_classes=3)
    
    # training loop
    for epoch in range(num_epochs):
        # Reset metrics for each epoch
        detection_metrics.reset()
        confusion_matrix.reset()

        model.train()
        train_accuracy = []
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            masks = batch['track'].to(device)
            
            optimizer.zero_grad()

            logits, depth_preds = model(images)

            # Compute losses
            loss_segmentation = criterion_segmentation(logits, masks)
            loss_depth = criterion_depth(depth_preds, depths)
            loss = loss_segmentation + loss_depth

            # Look into plotting individual losses for segmentation and depth to realize if one is is dominating the other
            loss.backward()
            optimizer.step()
            predicted_masks = logits.argmax(dim=1)

            # Compute IoU and depth metrics using DetectionMetrics
            detection_metrics.add(predicted_masks, masks, depth_preds, depths)
            metrics = detection_metrics.compute()
            # Compute confusion matrix for segmentation
            confusion_matrix.add(predicted_masks, masks)
            confusion_matrix_metrics = confusion_matrix.compute()

            train_accuracy.extend((predicted_masks == masks).cpu().detach().float().numpy())

            # Log images to TensorBoard
            grid = torchvision.utils.make_grid(images.cpu())
            logger.add_image("train/images", grid, global_step=global_step)
            
            # Log predicted masks to TensorBoard
            predicted_masks_grid = torchvision.utils.make_grid(predicted_masks.unsqueeze(1).cpu())
            logger.add_image("train/predicted_masks", predicted_masks_grid, global_step=global_step)

            logger.add_scalar("train/loss", loss.item(), global_step=global_step)

            global_step += 1

        logger.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)
    
        # Validation loop
        valid_accuracy = []
        model.eval()
        with torch.inference_mode():
            val_loss = 0
            for batch in val_loader:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                masks = batch['track'].to(device)

                logits, depth_preds = model(images)

                loss_segmentation = criterion_segmentation(logits, masks)
                loss_depth = criterion_depth(depth_preds, depths)

                val_loss += loss_segmentation.item() + loss_depth.item()

                predicted_masks = logits.argmax(dim=1)
                valid_accuracy.extend((predicted_masks == masks).cpu().detach().float().numpy())

            val_loss /= len(val_loader)
            logger.add_scalar("loss/val_total", val_loss, epoch)
            logger.add_scalar("val/accuracy", np.mean(valid_accuracy), epoch)

            logger.flush()

        # print epoch statistics
            print(
                f"Epoch {epoch + 1:2d} / {num_epochs:2d}: |"
                f"train_acc={np.mean(train_accuracy):.4f} |"
                f"val_acc={np.mean(valid_accuracy):.4f} | "
                f"abs_depth_error={metrics['abs_depth_error']:.4f} | "
                f"tp_depth_error={metrics['tp_depth_error']:.4f} |"
                f"IOU={confusion_matrix_metrics['iou']:.4f} | "
                f"accuracy={confusion_matrix_metrics['accuracy']:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=False, default="detector")
    parser.add_argument("--num_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-3)
    # parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=256)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
