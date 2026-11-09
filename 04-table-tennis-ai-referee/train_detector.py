"""Roboflow의 YOLO 라벨로 탁구공 검출기를 전이학습한다."""

import argparse
import os
from pathlib import Path
from tempfile import mkdtemp

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "datasets" / "table-tennis-ball-v2" / "data.yaml"
RUNS = ROOT / "runs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def check_split(images_dir: Path) -> tuple[Path | None, int, int]:
    """라벨 형식을 검사하고, 공이 있는 첫 이미지와 이미지 및 박스 수를 반환한다."""
    if not images_dir.is_dir() or images_dir.name != "images":
        raise ValueError(f"images 폴더를 찾을 수 없습니다. {images_dir}")
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not images:
        raise ValueError(f"학습에 사용할 이미지가 없습니다. {images_dir}")

    labels_dir = images_dir.with_name("labels")
    sample = None
    box_count = 0
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        if not label.is_file():
            continue  # YOLO에서는 라벨 파일이 없는 이미지를 배경으로 사용할 수 있다.

        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id, x, y, width, height = map(float, fields)
                if class_id != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError
            except ValueError:
                raise ValueError(f"잘못된 YOLO 박스입니다. {label}:{line_number} → {line}") from None
            box_count += 1
            if sample is None:
                sample = image
                print(f"  라벨 예시 {label.name} → {line}")
    return sample, len(images), box_count


def prepare_dataset(data_file: Path) -> tuple[dict, Path]:
    """Roboflow 내보내기의 경로를 확인하고 실행용 설정을 만든다."""
    if not data_file.is_file():
        raise ValueError(f"data.yaml이 없습니다. Roboflow ZIP을 풀고 파일 경로를 확인하세요. {data_file}")
    source = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict) or source.get("names") not in (["ball"], {0: "ball"}, {"0": "ball"}):
        raise ValueError("data.yaml의 0번 클래스가 ball인 단일 클래스 YOLO 데이터셋을 사용하세요.")

    config = {"path": str(data_file.parent), "names": ["ball"]}
    val_image = None
    for split in ("train", "val", "test"):
        entry = source.get(split)
        if split == "test" and not entry:
            continue
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"data.yaml에 {split} 이미지 폴더 경로가 필요합니다.")

        relative_path = Path(entry)
        images_dir = data_file.parent / relative_path
        # 일부 Roboflow ZIP은 같은 폴더 안의 train/images도 ../train/images로 표기한다.
        if relative_path.parts[:1] == ("..",):
            export_dir = data_file.parent.joinpath(*relative_path.parts[1:])
            if export_dir.is_dir():
                images_dir = export_dir
        if split == "test" and not images_dir.is_dir():
            print("test 이미지 폴더가 없어 테스트 분할은 생략합니다.")
            continue

        images_dir = images_dir.resolve()
        sample, image_count, box_count = check_split(images_dir)
        if sample is None and split != "test":
            raise ValueError(f"{split} 분할에 공 라벨이 하나도 없습니다. {images_dir}")
        print(f"{split}: 이미지 {image_count}장 / 공 박스 {box_count}개 ({images_dir})")
        config[split] = str(images_dir)
        if split == "val":
            val_image = sample

    if val_image is None:
        raise ValueError("검증 분할에 공이 표시된 이미지가 필요합니다.")
    if config["train"] == config["val"]:
        raise ValueError("train과 val은 서로 다른 이미지 폴더여야 합니다.")
    return config, val_image


def train(config: dict, val_image: Path, epochs: int, imgsz: int, batch: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU를 사용할 수 없습니다. README의 GPU 확인 명령을 먼저 실행하세요.")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    from ultralytics import YOLO

    RUNS.mkdir(exist_ok=True)
    run_dir = Path(mkdtemp(prefix="ball-", dir=RUNS))
    train_data = run_dir / "data.yaml"
    train_data.write_text(yaml.safe_dump(config), encoding="utf-8")
    print(f"GPU {torch.cuda.get_device_name(0)} · 학습 결과 {run_dir}", flush=True)

    # 강의 lab5와 같은 방식으로 COCO 사전학습 가중치를 공 1클래스에 맞춰 학습한다.
    model = YOLO(str(ROOT / "yolo11n.pt"))
    model.train(data=str(train_data), epochs=epochs, imgsz=imgsz, batch=batch,
                device=0, workers=0, project=str(RUNS), name=run_dir.name, exist_ok=True, seed=1729)

    best_path = run_dir / "weights" / "best.pt"
    best = YOLO(str(best_path))
    metrics = best.val(data=str(train_data), imgsz=imgsz, batch=batch, device=0, workers=0,
                       project=str(run_dir), name="validation", plots=False)
    print(f"검증 mAP50 {metrics.box.map50:.3f} / mAP50–95 {metrics.box.map:.3f}")
    result = best.predict(str(val_image), imgsz=imgsz, device=0, conf=0.25, verbose=False)[0]
    result.save(filename=str(run_dir / "val_prediction.jpg"))
    print(f"가중치 {best_path}\n예측 이미지 {run_dir / 'val_prediction.jpg'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="압축을 푼 Roboflow data.yaml")
    parser.add_argument("--check-only", action="store_true", help="데이터셋과 라벨만 검사")
    parser.add_argument("--epochs", type=int, default=1, help="학습 반복 횟수. 기본값은 연결 확인용 1회")
    parser.add_argument("--imgsz", type=int, default=640, help="모델 입력 이미지 크기")
    parser.add_argument("--batch", type=int, default=4, help="한 번에 학습할 이미지 수")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1:
        parser.error("epochs, imgsz, batch는 1 이상이어야 합니다.")

    try:
        config, val_image = prepare_dataset(args.data.resolve())
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"데이터셋 확인 실패. {error}\n")
    if not args.check_only:
        train(config, val_image, args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
