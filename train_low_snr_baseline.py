import argparse
import os
import time

import torch
import torch.nn.functional as F

from utils.config import dataset_path_dict
from utils.low_snr_baseline import (
    BASELINE_SPECS,
    add_awgn_torch,
    build_model,
    evaluate,
    make_train_val_loaders,
    run_root,
    save_checkpoint,
    set_reproducible_seed,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train fair supervised low-SNR baselines."
    )
    parser.add_argument("--baseline", choices=sorted(BASELINE_SPECS), required=True)
    parser.add_argument("--dataset", "-d", default="manytx")
    parser.add_argument("--epoch", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--awgn_min", type=float, default=-10.0)
    parser.add_argument("--awgn_max", type=float, default=20.0)
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--selection_snr_levels",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0],
    )
    return parser


def main(args):
    set_reproducible_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_path_dict[args.dataset]["pt_class"]
    spec = BASELINE_SPECS[args.baseline]
    lr = spec["lr"] if args.lr is None else args.lr
    output_root = run_root(args.baseline, args.dataset, args.random_seed)
    os.makedirs(output_root, exist_ok=True)

    train_loader, val_loader = make_train_val_loaders(
        args.dataset,
        args.batch_size,
        args.random_seed,
        args.num_workers,
    )
    model = build_model(args.baseline, num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epoch, 1)
    )
    metadata = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "seed": args.random_seed,
        "augmentation": spec["augmentation"],
        "epochs": args.epoch,
        "batch_size": args.batch_size,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "awgn_range": [args.awgn_min, args.awgn_max],
        "selection_snr_levels": args.selection_snr_levels,
    }
    write_json(os.path.join(output_root, "config.json"), metadata)

    best_acc = -1.0
    stale_epochs = 0
    start_time = time.time()
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        clean_correct = 0
        count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if spec["augmentation"] == "online_awgn":
                snr = torch.empty(inputs.shape[0], device=device).uniform_(
                    args.awgn_min, args.awgn_max
                )
                inputs = add_awgn_torch(inputs, snr)
            optimizer.zero_grad(set_to_none=True)
            if spec["augmentation"] == "paired_online_awgn":
                snr = torch.empty(inputs.shape[0], device=device).uniform_(
                    args.awgn_min, args.awgn_max
                )
                noisy_inputs = add_awgn_torch(inputs, snr)
                paired_logits = model(torch.cat([inputs, noisy_inputs], dim=0))
                clean_logits, noisy_logits = paired_logits.chunk(2, dim=0)
                loss = 0.5 * (
                    F.cross_entropy(clean_logits, targets)
                    + F.cross_entropy(noisy_logits, targets)
                )
                logits = noisy_logits
                clean_correct += (
                    clean_logits.argmax(dim=1) == targets
                ).sum().item()
            else:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                clean_correct += (logits.argmax(dim=1) == targets).sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.shape[0]
        scheduler.step()

        val_clean = evaluate(model, val_loader, device)
        val_low_snr = []
        for snr_index, snr in enumerate(args.selection_snr_levels):
            set_reproducible_seed(
                args.random_seed + epoch * len(args.selection_snr_levels) + snr_index
            )
            val_low_snr.append(evaluate(model, val_loader, device, snr=snr))
        selection_acc = sum(item["acc"] for item in val_low_snr) / len(val_low_snr)
        selection_f1 = sum(item["macro_f1"] for item in val_low_snr) / len(val_low_snr)
        if spec["augmentation"] == "paired_online_awgn":
            train_metrics = (
                f"train_noisy_acc={100.0 * correct / max(count, 1):.2f}%, "
                f"train_clean_acc={100.0 * clean_correct / max(count, 1):.2f}%"
            )
        else:
            train_metrics = f"train_acc={100.0 * correct / max(count, 1):.2f}%"
        print(
            f"Epoch {epoch + 1}/{args.epoch}: "
            f"loss={total_loss / max(count, 1):.6f}, "
            f"{train_metrics}, "
            f"val_clean_acc={val_clean['acc']:.2f}%, "
            f"val_low_snr_acc={selection_acc:.2f}%, "
            f"val_low_snr_f1={selection_f1:.2f}%",
            flush=True,
        )
        if selection_acc > best_acc:
            best_acc = selection_acc
            stale_epochs = 0
            save_checkpoint(
                os.path.join(output_root, "best.pth"), model, metadata
            )
            print(f"Best checkpoint saved at epoch {epoch + 1}.", flush=True)
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}.", flush=True)
            break

    save_checkpoint(os.path.join(output_root, "final.pth"), model, metadata)
    print(
        f"End. best_val_low_snr_acc={best_acc:.2f}%, "
        f"minutes={(time.time() - start_time) / 60.0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main(build_parser().parse_args())
