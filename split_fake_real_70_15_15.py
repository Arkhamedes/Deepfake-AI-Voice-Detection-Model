import argparse
import random
import shutil
from pathlib import Path


def collect_files(class_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(class_dir.glob(f"*{ext}"))
    return sorted(files)


def split_indices(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_and_rename(
    files: list[Path],
    split_name: str,
    class_name: str,
    dest_root: Path,
) -> None:
    dest_dir = dest_root / split_name / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    class_tag = class_name.lower()
    for i, src in enumerate(files, start=1):
        new_name = f"{split_name}_{class_tag}_{i:06d}{src.suffix.lower()}"
        dst = dest_dir / new_name
        shutil.copy2(src, dst)


def split_class(
    class_name: str,
    src_root: Path,
    dest_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    extensions: tuple[str, ...],
) -> dict[str, int]:
    class_dir = src_root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    files = collect_files(class_dir, extensions)
    if not files:
        raise ValueError(f"No files found for class {class_name} in {class_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    train_count, val_count, test_count = split_indices(len(files), train_ratio, val_ratio)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    copy_and_rename(train_files, "train", class_name, dest_root)
    copy_and_rename(val_files, "val", class_name, dest_root)
    copy_and_rename(test_files, "test", class_name, dest_root)

    return {
        "total": len(files),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split FAKE/REAL data into class-balanced 70/15/15 train/val/test sets."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("5s_clips") / "AUDIO_CLEAN",
        help="Source directory containing FAKE and REAL folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Final_Merged_Dataset") / "split_70_15_15",
        help="Output root directory for split data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp3"],
        help="File extensions to include, e.g. .mp3 .wav",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory before writing split files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = args.source
    output = args.output

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1")

    if output.exists() and args.overwrite:
        shutil.rmtree(output)

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output}. Use --overwrite to replace it."
        )

    classes = ["FAKE", "REAL"]
    exts = tuple(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions)

    summary: dict[str, dict[str, int]] = {}
    for class_name in classes:
        summary[class_name] = split_class(
            class_name=class_name,
            src_root=source,
            dest_root=output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            extensions=exts,
        )

    print("Split completed.")
    print(f"Source: {source}")
    print(f"Output: {output}")
    for class_name in classes:
        stats = summary[class_name]
        print(
            f"{class_name}: total={stats['total']} train={stats['train']} val={stats['val']} test={stats['test']}"
        )


if __name__ == "__main__":
    main()
