import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_age(age: float, min_age: float, max_age: float) -> float:
    x = (age - min_age) / (max_age - min_age)
    return max(0.0, min(1.0, x))


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_image(src: Path, size: int) -> Image.Image:
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = center_crop_square(img)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Root folder with age subfolders")
    parser.add_argument("--output-root", required=True, help="Output dataset folder")
    parser.add_argument("--size", type=int, default=512, help="Output image size")
    parser.add_argument("--min-age", type=float, default=10.0)
    parser.add_argument("--max-age", type=float, default=80.0)
    parser.add_argument("--max-per-age", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = []
    global_idx = 0

    age_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    for age_dir in age_dirs:
        try:
            age = int(age_dir.name)
        except ValueError:
            print(f"Skipping non-age folder: {age_dir}")
            continue

        files = sorted(
            [p for p in age_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
        )

        if args.max_per_age > 0:
            files = files[:args.max_per_age]

        age_norm = normalize_age(age, args.min_age, args.max_age)

        for src in files:
            try:
                img = process_image(src, args.size)
            except Exception as e:
                print(f"Failed on {src}: {e}")
                continue

            subdir = f"{global_idx // 1000:05d}"
            out_dir = output_root / subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"img{global_idx:08d}.png"
            out_path = out_dir / out_name
            img.save(out_path, format="PNG", compress_level=4)

            rel_path = out_path.relative_to(output_root).as_posix()
            labels.append([rel_path, [float(age_norm)]])

            global_idx += 1

    dataset = {"labels": labels}
    with open(output_root / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Done.")
    print(f"Images written: {global_idx}")
    print(f"Dataset path: {output_root}")
    print(f"Labels file: {output_root / 'dataset.json'}")


if __name__ == "__main__":
    main()