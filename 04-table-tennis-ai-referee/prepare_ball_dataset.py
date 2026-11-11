"""다운로드한 세 종류의 공 라벨을 한 클래스 학습용 사본으로 준비한다."""
from collections import Counter
from pathlib import Path
import hashlib
import re
import zipfile

import yaml

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / 'datasets'
OUTPUT = DATASETS / 'ball-combined-v1'
SOURCES = [
    ('relabeling-ism24-v1-yolov11.zip', ['ball', 'net', 'player', 'racket', 'referee', 'scoreboard', 'table']),
    ('table-tennis-ball-tracking-v1-yolov11.zip', ['Tennis_ball']),
    ('table-tennis-ball-5rw9y-v1-yolov11.zip', ['negative-slope', 'other', 'positive-slope', 'zero-slope']),
]


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw/2, bx + bw/2) - max(ax - aw/2, bx - bw/2))
    ih = max(0, min(ay + ah/2, by + bh/2) - max(ay - ah/2, by - bh/2))
    intersection = iw * ih
    return intersection / (aw*ah + bw*bh - intersection)


def main():
    OUTPUT.mkdir(exist_ok=False)
    counts = Counter()
    seen = set()
    for archive, expected_classes in SOURCES:
        with zipfile.ZipFile(DATASETS / archive) as z:
            data = yaml.safe_load(z.read('data.yaml'))
            if data['names'] != expected_classes:
                raise ValueError(f'예상과 다른 클래스: {archive} {data["names"]}')
            for image_name in sorted(n for n in z.namelist() if '/images/' in n and n.lower().endswith(('.jpg', '.jpeg', '.png'))):
                split = image_name.split('/')[0]
                label_name = re.sub(r'\.[^.]+$', '.txt', image_name.replace('/images/', '/labels/'))
                if label_name not in z.namelist():
                    raise ValueError(f'라벨 파일 없음: {label_name}')
                boxes = []
                for line in z.read(label_name).decode('utf-8-sig').splitlines():
                    values = line.split()
                    if not values:
                        continue
                    class_id = int(values[0])
                    if not 0 <= class_id < len(expected_classes):
                        raise ValueError(f'클래스 번호 오류: {label_name} {line}')
                    if archive.startswith('relabeling') and class_id != 0:
                        continue
                    if len(values) != 5:
                        raise ValueError(f'박스 형식 오류: {label_name} {line}')
                    box = tuple(map(float, values[1:]))
                    x, y, w, h = box
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        raise ValueError(f'박스 좌표 오류: {label_name} {line}')
                    if any(overlap(box, prior) >= .5 for prior in boxes):
                        counts['overlapping boxes removed'] += 1
                    else:
                        boxes.append(box)
                if not boxes:
                    # 공이 라벨링되지 않았다고 해서 공이 없다는 뜻은 아니다.
                    counts['unlabeled images excluded'] += 1
                    continue
                content = z.read(image_name)
                digest = hashlib.sha256(content).hexdigest()
                if digest in seen:
                    counts['duplicate images excluded'] += 1
                    continue
                seen.add(digest)
                if 'tracking' in archive:
                    # 같은 동영상의 연속 프레임이 서로 다른 분할에 섞이지 않게 한다.
                    split = 'train'
                elif '5rw9y' in archive:
                    match = re.search(r'game0_point(00|01)_', image_name)
                    if not match:
                        raise ValueError(f'랠리 그룹 확인 실패: {image_name}')
                    split = 'train' if match.group(1) == '00' else 'valid'
                if split not in ('train', 'valid', 'test'):
                    raise ValueError(f'알 수 없는 분할: {image_name}')
                for folder in ('images', 'labels'):
                    (OUTPUT / split / folder).mkdir(parents=True, exist_ok=True)
                suffix = Path(image_name).suffix.lower()
                (OUTPUT / split / 'images' / (digest + suffix)).write_bytes(content)
                (OUTPUT / split / 'labels' / (digest + '.txt')).write_text(
                    ''.join('0 ' + ' '.join(map(str, box)) + '\n' for box in boxes), encoding='utf-8')
                counts[f'{split} images'] += 1
                counts[f'{split} ball boxes'] += len(boxes)
    config = {'path': str(OUTPUT), 'train': 'train/images', 'val': 'valid/images',
              'test': 'test/images', 'names': ['ball']}
    (OUTPUT / 'data.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    print('\n'.join(f'{key}: {value}' for key, value in sorted(counts.items())))
    print('dataset:', OUTPUT)


if __name__ == '__main__':
    main()
