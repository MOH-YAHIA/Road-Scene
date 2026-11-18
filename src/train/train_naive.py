import logging
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.camvid_dataset import CamVidDataset
from src.models.naive_segmentation import NaiveSegmentationModel
from src.utils.logger import setup_logger
from src.utils.train_utils import run_epoch


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config_path = Path("configs/niave_config.yml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    setup_logger(config["output"]["log_dir"])
    logger = logging.getLogger(__name__)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.critical("Device: %s", device)

    # Datasets
    dataset_config = config["dataset"]

    train_dataset = CamVidDataset(
        root_dir=dataset_config["root_dir"],
        split="train",
        image_size=tuple(dataset_config["image_size"]),
        ignore_index=dataset_config["ignore_index"],
    )

    val_dataset = CamVidDataset(
        root_dir=dataset_config["root_dir"],
        split="val",
        image_size=tuple(dataset_config["image_size"]),
        ignore_index=dataset_config["ignore_index"],
    )

    num_classes = train_dataset.num_classes

    configured_classes = config["model"]["num_classes"]
    if configured_classes is not None and configured_classes != num_classes:
        logger.warning(
            "Configured classes %d but dataset has %d classes",
            num_classes,
            configured_classes,
        )

    logger.info("Classes (%d): %s", num_classes, train_dataset.classes)
    logger.info("Train images: %d", len(train_dataset))
    logger.info("Validation images: %d", len(val_dataset))

    loader_config = config["training"]
    use_cuda_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=True,
        num_workers=loader_config["num_workers"],
        pin_memory=use_cuda_memory,
        persistent_workers=loader_config["num_workers"] > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=False,
        num_workers=loader_config["num_workers"],
        pin_memory=use_cuda_memory,
        persistent_workers=loader_config["num_workers"] > 0,
    )

    # Model
    model_config = config["model"]

    model = NaiveSegmentationModel(
        num_classes=num_classes,
        n_freeze_layers=model_config["n_freeze_layers"],
    ).to(device)

    # CrossEntropyLoss expects raw logits and integer class masks.
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_config["ignore_index"])

    # Use different learning rates for pretrained and new layers.
    optimizer = AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": loader_config["encoder_lr"],
            },
            {
                "params": model.decoder.parameters(),
                "lr": loader_config["decoder_lr"],
            },
        ],
        weight_decay=loader_config["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=loader_config["scheduler_factor"],
        patience=loader_config["scheduler_patience"],
        min_lr=loader_config["min_lr"],
    )

    output_config = config["output"]
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / output_config["best_model_name"]
    last_path = checkpoint_dir / output_config["last_model_name"]

    best_miou = -1.0

    logger.info("Starting training for %d epochs", loader_config["epochs"])

    for epoch in range(1, loader_config["epochs"] + 1):
        train_metrics = run_epoch(
            model=model,
            n_freeze_layers=model_config["n_freeze_layers"],
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=dataset_config["ignore_index"],
            optimizer=optimizer,
            logger=logger,
            mode="train",
        )

        val_metrics = run_epoch(
            model=model,
            n_freeze_layers=model_config["n_freeze_layers"],
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=dataset_config["ignore_index"],
            logger=logger,
            mode="val",
        )

        scheduler.step(val_metrics["miou"])

        current_lrs = [group["lr"] for group in optimizer.param_groups]

        logger.info(
            "Epoch %03d/%03d | "
            "Train loss: %.4f, pixel acc: %.4f, mIoU: %.4f | "
            "Val loss: %.4f, pixel acc: %.4f, mIoU: %.4f | "
            "Encoder LR: %.2e, Decoder LR: %.2e",
            epoch,
            loader_config["epochs"],
            train_metrics["loss"],
            train_metrics["pixel_accuracy"],
            train_metrics["miou"],
            val_metrics["loss"],
            val_metrics["pixel_accuracy"],
            val_metrics["miou"],
            current_lrs[0],
            current_lrs[1],
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_miou": val_metrics["miou"],
            "best_val_miou": max(best_miou, val_metrics["miou"]),
            "num_classes": num_classes,
            "classes": train_dataset.classes,
            "config": config,
        }

        torch.save(checkpoint, last_path)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            checkpoint["best_val_miou"] = best_miou
            torch.save(checkpoint, best_path)

            logger.info(
                "New best model saved: %s (val mIoU: %.4f)",
                best_path,
                best_miou,
            )

    logger.info("Training finished.")
    logger.info("Best validation mIoU: %.4f", best_miou)
    logger.info("Best checkpoint: %s", best_path)
    logger.info("Last checkpoint: %s", last_path)


if __name__ == "__main__":
    main()
