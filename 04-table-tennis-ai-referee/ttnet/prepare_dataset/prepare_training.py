"""Extract labelled training samples from downloaded OpenTTGames videos."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2


def prepare(root, game):
    dataset = root / 'dataset' / 'training'
    archive = dataset / 'annotations' / f'{game}.zip'
    annotations = dataset / 'annotations' / game
    video = dataset / 'videos' / f'{game}.mp4'
    images = dataset / 'images' / game
    if not archive.is_file() or not video.is_file():
        raise FileNotFoundError(f'Missing video or annotation archive for {game}')
    annotations.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Reject archive paths outside the intended annotations directory.
        for member in z.infolist():
            if not (annotations / member.filename).resolve().is_relative_to(annotations.resolve()):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        for member in z.infolist():
            target = annotations / member.filename
            if not target.exists():
                z.extract(member, annotations)
    balls = json.loads((annotations / 'ball_markup.json').read_text(encoding='utf-8'))
    events = json.loads((annotations / 'events_markup.json').read_text(encoding='utf-8'))
    frames = set()
    for idx in map(int, events):
        ball = balls.get(str(idx + 4))
        mask = annotations / 'segmentation_masks' / f'{idx + 4}.png'
        if idx >= 4 and ball and ball['x'] >= 0 and ball['y'] >= 0 and mask.is_file():
            frames.update(range(idx - 4, idx + 5))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video}')
    missing = sorted(i for i in frames if not (images / f'img_{i:06d}.jpg').is_file())
    print(f'{game}: {len(frames)} frames required, {len(missing)} to extract', flush=True)
    try:
        previous = -2
        for count, idx in enumerate(missing, 1):
            if idx != previous + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read {game} frame {idx}')
            if not cv2.imwrite(str(images / f'img_{idx:06d}.jpg'), frame):
                raise RuntimeError(f'Cannot save {game} frame {idx}')
            previous = idx
            if count % 500 == 0:
                print(f'{game}: {count}/{len(missing)} frames', flush=True)
    finally:
        cap.release()
    print(f'{game}: preparation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('games', nargs='+', help='For example: game_1 game_2')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    for name in args.games:
        if name not in {f'game_{i}' for i in range(1, 6)}:
            parser.error(f'Unknown game: {name}')
        prepare(root, name)
