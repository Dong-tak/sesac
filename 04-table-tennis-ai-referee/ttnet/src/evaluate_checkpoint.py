"""Evaluate a trusted TTNet checkpoint on a separate annotated game."""

import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from data_process.ttnet_data_utils import get_events_infor
from data_process.ttnet_dataset import TTNet_Dataset
from models.model_utils import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--game', default='game_5')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    # The training checkpoint contains Python objects and must come from a trusted source.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['configs']
    if args.game in config.train_game_list:
        raise ValueError('Evaluation game was used for training')
    config.dataset_dir = './dataset'
    config.device = torch.device('cuda:0')
    config.smooth_labelling = False
    examples, _ = get_events_infor([args.game], config, 'training')
    if not examples:
        raise ValueError('No annotated examples found')
    loader = DataLoader(TTNet_Dataset(examples, config.org_size, config.input_size),
                        batch_size=1, shuffle=False, num_workers=0)
    model = create_model(config).to(config.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    width, height = config.input_size
    event_counts = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in ('bounce', 'net')}
    threshold_counts = {str(t): {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in ('bounce', 'net')}
                        for t in (0.1, 0.2, 0.3, 0.4, 0.5)}
    distances = []
    local_distances = []
    missed = 0
    with torch.inference_mode():
        for sequence, original_ball, ball, target_events, _ in loader:
            global_ball, local_ball, events, _ = model.run_demo(sequence.float().to(config.device))
            pred = global_ball[0].cpu()
            if pred[:width].max() < config.thresh_ball_pos_mask or pred[width:].max() < config.thresh_ball_pos_mask:
                missed += 1
            else:
                x = int(pred[:width].argmax())
                y = int(pred[width:].argmax())
                dx = (x - int(ball[0, 0])) / width
                dy = (y - int(ball[0, 1])) / height
                distances.append(math.hypot(dx, dy))
                local_pred = local_ball[0].cpu()
                x_min = max(0, x * 1920 // width - width // 2)
                y_min = max(0, int(y * 1080 / height) - height // 2)
                x_pad = (width - (min(1920, x_min + width) - x_min)) // 2
                y_pad = (height - (min(1080, y_min + height) - y_min)) // 2
                lx = x_min + int(local_pred[:width].argmax()) - x_pad
                ly = y_min + int(local_pred[width:].argmax()) - y_pad
                local_distances.append(math.hypot((lx - int(original_ball[0, 0])) / 1920,
                                                  (ly - int(original_ball[0, 1])) / 1080))
            for i, name in enumerate(('bounce', 'net')):
                predicted = events[0, i].item() >= 0.5
                actual = target_events[0, i].item() > 0
                event_counts[name]['tp'] += int(predicted and actual)
                event_counts[name]['fp'] += int(predicted and not actual)
                event_counts[name]['fn'] += int(not predicted and actual)
                for threshold, per_event in threshold_counts.items():
                    predicted_at_threshold = events[0, i].item() >= float(threshold)
                    per_event[name]['tp'] += int(predicted_at_threshold and actual)
                    per_event[name]['fp'] += int(predicted_at_threshold and not actual)
                    per_event[name]['fn'] += int(not predicted_at_threshold and actual)
    for counts in event_counts.values():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        counts['precision'] = tp / (tp + fp) if tp + fp else None
        counts['recall'] = tp / (tp + fn) if tp + fn else None
    result = {'checkpoint': args.checkpoint, 'game': args.game, 'samples': len(examples),
              'event_threshold': 0.5, 'events': event_counts, 'threshold_counts': threshold_counts,
              'ball_missed': missed,
              'ball_mean_normalized_error': sum(distances) / len(distances) if distances else None,
              'ball_within_2_percent': sum(d <= 0.02 for d in distances),
              'ball_within_5_percent': sum(d <= 0.05 for d in distances),
              'display_ball_mean_normalized_error': sum(local_distances) / len(local_distances) if local_distances else None,
              'display_ball_within_2_percent': sum(d <= 0.02 for d in local_distances),
              'display_ball_within_5_percent': sum(d <= 0.05 for d in local_distances)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
