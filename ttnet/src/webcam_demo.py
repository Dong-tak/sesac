"""Run a locally trained TTNet checkpoint on a webcam. Press Q to quit."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from models.model_utils import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Path to your trusted TTNet checkpoint')
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default: 0)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available in this Python environment')
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    # TTNet training checkpoints include Python objects; only open trusted files.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    configs = checkpoint['configs']
    configs.device = torch.device('cuda:0')
    model = create_model(configs).to(configs.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open webcam {args.camera}; check connection and camera permissions')
    frames = deque(maxlen=configs.num_frames_sequence)
    resized = deque(maxlen=configs.num_frames_sequence)
    width, height = configs.input_size
    print('Webcam running on CUDA. Press Q in the video window to exit.', flush=True)
    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Failed to read webcam frame')
                frames.append(frame)
                resized.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height)))
                if len(frames) < configs.num_frames_sequence:
                    cv2.imshow('TTNet webcam - Q to quit', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                sequence = np.dstack(resized).transpose(2, 0, 1).copy()
                inputs = torch.from_numpy(sequence).unsqueeze(0).float().to(configs.device)
                global_ball, local_ball, events, segmentation = model.run_demo(inputs)
                # Ball labels in this project refer to the last frame, while event labels refer to the middle frame.
                display = frames[-1].copy()
                h, w = display.shape[:2]
                global_ball = global_ball[0].cpu().numpy()
                local_ball = local_ball[0].cpu().numpy()
                gx, gy = int(np.argmax(global_ball[:width])), int(np.argmax(global_ball[width:]))
                lx, ly = int(np.argmax(local_ball[:width])), int(np.argmax(local_ball[width:]))
                if (global_ball[:width].max() >= configs.thresh_ball_pos_mask and
                        global_ball[width:].max() >= configs.thresh_ball_pos_mask):
                    # Match TTNet's fixed 1920x1080 local crop before scaling to the webcam image.
                    x_min = max(0, gx * 1920 // width - width // 2)
                    y_min = max(0, int(gy * 1080 / height) - height // 2)
                    x_pad = (width - (min(1920, x_min + width) - x_min)) // 2
                    y_pad = (height - (min(1080, y_min + height) - y_min)) // 2
                    x = int((x_min + lx - x_pad) * w / 1920)
                    y = int((y_min + ly - y_pad) * h / 1080)
                    cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
                scores = events[0].cpu().numpy()
                cv2.putText(display, f'Bounce (4 frames ago): {scores[0]:.2f}  Net: {scores[1]:.2f}  |  Q: quit',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('TTNet webcam - Q to quit', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
