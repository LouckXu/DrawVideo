import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List


class SketchConverter:
    """Convert a colour image to a black-and-white line sketch via colour-dodge blending."""

    def __init__(self):
        self.track1 = None
        self.track2 = None

    def imread(self, path: Path):
        """Read an image using numpy byte-buffer to handle arbitrary file paths safely."""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print("READ ERROR:", path, e)
            return None

    def step1(self, frame):
        """Convert BGR frame to greyscale; return two independent copies."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return g.copy(), g.copy()

    def step2(self):
        """Invert the first grey channel."""
        return 255 - self.track1

    def step3(self):
        """Erode the inverted channel with a 3x3 kernel."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(self.track1, kernel, iterations=1)

    def step4(self):
        """Apply colour-dodge blend between the two channels to produce line art."""
        t1 = self.track1.astype(np.float32)
        t2 = self.track2.astype(np.float32)

        mask = t1 < 255
        out = np.zeros_like(t2)
        out[mask] = np.clip((t2[mask] * 255.0) / (255.0 - t1[mask]), 0, 255)
        out[~mask] = 255
        return out.astype(np.uint8)

    def process(self, frame):
        """Run the full sketch extraction pipeline and return a greyscale uint8 array."""
        self.track1, self.track2 = self.step1(frame)
        self.track1 = self.step2()
        self.track1 = self.step3()
        return self.step4()


def find_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert colour keyframe images to black-and-white line sketches."
    )
    parser.add_argument("input", type=str, help="Path to input folder containing images.")
    parser.add_argument("-o", "--output", type=str, default="output_sketch",
                        help="Output folder for sketch PNG files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    images = find_images(input_path)
    print("Found images:", len(images))
    print("Output:", output_path.resolve())

    converter = SketchConverter()
    success = 0
    failed = 0

    for i, img_path in enumerate(sorted(images), 1):
        img = converter.imread(img_path)
        if img is None:
            print("[SKIP READ FAIL]", img_path)
            failed += 1
            continue

        try:
            result = converter.process(img)
            save_path = output_path / f"{img_path.stem}_sketch.png"
            Image.fromarray(result).save(save_path)
            print(f"[{i}/{len(images)}] Saved:", save_path.name)
            success += 1
        except Exception as e:
            print("[ERROR]", img_path, e)
            failed += 1

    print(f"\nDone. Success: {success}  Failed: {failed}")
    print("Output folder:", output_path.resolve())


if __name__ == "__main__":
    main()
