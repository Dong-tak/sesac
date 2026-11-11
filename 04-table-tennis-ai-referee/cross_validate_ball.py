"""영상·랠리 묶음을 분리해 공 검출 모델을 네 번 평가한다."""
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
import hashlib
import json
import os
import re
import zipfile

import yaml

from prepare_ball_dataset import DATASETS, OUTPUT, ROOT, SOURCES


def image_groups():
    groups = {}
    for archive, _ in SOURCES:
        with zipfile.ZipFile(DATASETS / archive) as z:
            for name in z.namelist():
                if '/images/' not in name or not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                digest = hashlib.sha256(z.read(name)).hexdigest()
                if archive.startswith('relabeling'):
                    group = 'relabeling_unknown_origins'
                elif 'tracking' in archive:
                    group = 'tracking_video'
                else:
                    match = re.search(r'game0_point(00|01)_', name)
                    if not match:
                        raise ValueError(f'랠리 그룹 확인 실패: {name}')
                    group = f'game0_point{match.group(1)}'
                if digest in groups and groups[digest] != group:
                    raise ValueError(f'서로 다른 그룹에서 중복 이미지 발견: {name}')
                groups[digest] = group
    return groups


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU를 사용할 수 없습니다.')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
    from ultralytics import YOLO

    groups = image_groups()
    images = sorted(p for split in ('train', 'valid') for p in (OUTPUT / split / 'images').iterdir() if p.is_file())
    if not images or any(p.stem not in groups for p in images):
        raise ValueError('이미지 원본 그룹을 확인할 수 없습니다.')
    group_names = sorted({groups[p.stem] for p in images})
    if len(group_names) != 4:
        raise ValueError(f'그룹 수 확인 실패: {group_names}')

    (ROOT / 'runs').mkdir(exist_ok=True)
    run = Path(mkdtemp(prefix='ball-cv-', dir=ROOT / 'runs'))
    print('group sizes:', dict(Counter(groups[p.stem] for p in images)), flush=True)
    results = []
    for i, group in enumerate(group_names, 1):
        train = [p for p in images if groups[p.stem] != group]
        valid = [p for p in images if groups[p.stem] == group]
        train_list, valid_list = run / f'fold{i}-train.txt', run / f'fold{i}-valid.txt'
        train_list.write_text(''.join(str(p) + '\n' for p in train), encoding='utf-8')
        valid_list.write_text(''.join(str(p) + '\n' for p in valid), encoding='utf-8')
        config = run / f'fold{i}.yaml'
        config.write_text(yaml.safe_dump({'path': str(OUTPUT), 'train': str(train_list),
                                          'val': str(valid_list), 'names': ['ball']}), encoding='utf-8')
        print(f'FOLD {i}/4 group={group} train={len(train)} val={len(valid)}', flush=True)
        model = YOLO(str(ROOT / 'yolo11n.pt'))
        model.train(data=str(config), epochs=30, imgsz=640, batch=4, device=0, workers=0,
                    project=str(run), name=f'fold{i}', seed=1729,
                    optimizer='AdamW', lr0=0.002, momentum=0.9,
                    mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                    translate=0.1, scale=0.5, close_mosaic=10)
        best_path = run / f'fold{i}' / 'weights' / 'best.pt'
        metrics = YOLO(str(best_path)).val(data=str(config), imgsz=640, batch=4,
                                           device=0, workers=0, plots=False,
                                           project=str(run / f'fold{i}'), name='best-validation')
        results.append({'fold': i, 'validation_group': group, 'train_images': len(train),
                        'validation_images': len(valid), 'map50': float(metrics.box.map50),
                        'map50_95': float(metrics.box.map), 'weights': str(best_path)})
        (run / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print('FOLD RESULT:', json.dumps(results[-1]), flush=True)
    print('MEAN mAP50:', sum(r['map50'] for r in results)/len(results), flush=True)
    print('MEAN mAP50-95:', sum(r['map50_95'] for r in results)/len(results), flush=True)
    print('RESULT DIRECTORY:', run, flush=True)


if __name__ == '__main__':
    main()
