import torch


def calculate_metrics(confusion_matrix):
    """confusion_matrix[true_class, predicted_class]"""
    confusion_matrix = confusion_matrix.float()

    true_positive = confusion_matrix.diag()
    ground_truth_count = confusion_matrix.sum(dim=1)
    predicted_count = confusion_matrix.sum(dim=0)

    union = ground_truth_count + predicted_count - true_positive

    valid_classes = union > 0
    iou = true_positive[valid_classes] / union[valid_classes]

    miou = iou.mean().item() if iou.numel() else 0.0

    total_pixels = confusion_matrix.sum()
    accuracy = (true_positive.sum() / total_pixels).item() if total_pixels > 0 else 0.0

    return accuracy, miou


def update_confusion_matrix(
    confusion_matrix, logits, masks, num_classes, ignore_index=255
):
    predictions = logits.argmax(dim=1)
    valid = masks != ignore_index

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = ((masks == true_class) & (predictions == pred_class) & valid).sum()

            confusion_matrix[true_class, pred_class] += count


def run_epoch(
    model,
    n_freeze_layers,
    loader,
    criterion,
    device,
    num_classes,
    ignore_index,
    optimizer=None,
    logger=None,
    mode="train",
):
    training = optimizer is not None
    model.train(training)
    if training and n_freeze_layers > 0:
        for layer in model.encoder.backbone[:n_freeze_layers]:
            layer.eval()

    total_loss = 0.0
    total_images = 0

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64,
        device=device,
    )

    for idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        # No gradients are computed during validation.
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_images += batch_size

        update_confusion_matrix(
            confusion_matrix,
            logits.detach(),
            masks,
            num_classes,
            ignore_index,
        )

        if mode == "train" and logger is not None and idx % (len(loader) // 5) == 0:
            logger.info(
                "Batch %d/%d | Loss: %.4f ", idx + 1, len(loader), loss.detach().item()
            )
    average_loss = total_loss / max(total_images, 1)
    pixel_accuracy, miou = calculate_metrics(confusion_matrix)

    return {
        "loss": average_loss,
        "pixel_accuracy": pixel_accuracy,
        "miou": miou,
    }
