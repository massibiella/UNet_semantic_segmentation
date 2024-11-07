import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.utils.tensorboard as tb

from .models import Classifier, load_model, save_model
from .datasets.classification_dataset import load_data

def train(
        exp_dir: str = "logs",
        model_name: str = "classifier",
        num_epochs: int = 50,
        lr: float = 1e-2,
        batch_size: int = 512,
        seed: int = 2024,
        **kwargs
):
    
    # set random seed so each run is deterministic
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # load data  
    train_loader = load_data("classification_data/train", transform_pipeline="aug",  return_dataloader=True, num_workers=0, batch_size=batch_size, shuffle=True)
    val_loader = load_data("classification_data/val", return_dataloader=True, num_workers=0, batch_size=batch_size, shuffle=False)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name)
    model = model.to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0


    # training loop
    for epoch in range(num_epochs):

        model.train()
        train_accuracy = []
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = torch.nn.functional.cross_entropy(pred, label)

            train_accuracy.extend((pred.argmax(dim=-1) == label).cpu().detach().float().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log loss to tensorboard
            logger.add_scalar("train/loss", loss.item(), global_step=global_step)
            global_step += 1

        logger.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)
    
        # disable gradient computation and switch to evaluation mode
        valid_accuracy = []
        model.eval()
        with torch.inference_mode():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                # Compute validation accuracy
                valid_accuracy.extend((pred.argmax(dim=-1) == label).cpu().detach().float().numpy())

            logger.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)

            logger.flush()

        # print on first, last, every 10th epoch:
            print(
                f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
                f"train_acc={np.mean(train_accuracy):.4f} "
                f"val_acc={np.mean(valid_accuracy):.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-2)
    # parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=512)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
