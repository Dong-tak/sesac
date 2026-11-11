# 탁구공 검출기 학습 실습

YOLO 특강 5차시의 전이학습 실습(`lab5_yolo11.py`) 순서를 따른다. 라벨을 확인하고, COCO 사전학습 모델 `yolo11n.pt`를 탁구공 한 클래스로 전이학습한 뒤, `best.pt`로 검증한다. 강의는 `yolo11s.pt`를 사용했다. 여기서는 PC 실시간 추론을 고려해 더 작은 `yolo11n.pt`로 시작한다. 득점 모델과 실시간 영상 연결은 [탁구 AI 심판 기획서](PLAN.md)에 정리했다.

## 2026-11-11 데이터 실험과 주의사항

`prepare_ball_dataset.py`는 네 Roboflow 프로젝트 중 공 라벨이 있는 세 프로젝트의 ZIP을 읽어 공 한 클래스 학습용 사본을 만든다. 탁구대 전용 87장은 다른 공 프로젝트의 이미지와 동일하여 별도로 넣지 않는다. `cross_validate_ball.py`는 사본에서 4개 영상·랠리 묶음별 교차 검증을 수행한다. `train_detector.py`와 교차 검증 코드는 다음 실행부터 AdamW를 명시하고, 모자이크·반전·색상 및 위치 변화 등 증강은 학습에 적용한다.

**라벨 주의:** 최초 다운로드한 `reviewed-100-yolov11.zip`은 100장 모두 공 박스 하나였으나, 나중에 받은 `relabeling-ism24-v1-yolov11.zip`에서는 84장에 공 박스가 두 개였다. 아래 준비 스크립트는 후자의 80장에서 IoU 0.5 이상인 뒤쪽 박스를 자동 제외하며, 이는 원본 정답을 복원했다는 보장이 없다. 나머지 4장은 두 박스를 유지한다. 다음 실험 전에는 최초 ZIP과 Roboflow 라벨을 비교·검수하고, 공 외 객체 라벨이 빠진 데이터는 여러 객체 검출에 무턱대고 합치지 않는다. 최초 ZIP은 이 스크립트의 입력이 아니다.

실험용 ZIP과 모델 가중치, 결과 이미지 등은 Git에 포함하지 않는다. 로컬 `datasets/` 폴더에 스크립트에 명시된 세 ZIP을 준비한 경우에만 다음 명령을 실행한다. 결과 폴더가 이미 있으면 덮어쓰지 않는다.

```powershell
uv run python -X utf8 prepare_ball_dataset.py
uv run python -X utf8 train_detector.py --data datasets/ball-combined-v1/data.yaml --check-only
uv run python -X utf8 train_detector.py --data datasets/ball-combined-v1/data.yaml --epochs 30
uv run python -X utf8 cross_validate_ball.py
```

교차 검증에는 학습·검증 193장을 사용했고 별도의 테스트 10장은 사용하지 않았다. 폴드별 mAP50은 point00 0.415, point01 0.424, relabeling 0.026, tracking 0.175였다. 평균은 0.260이다. 두 랠리는 같은 경기이고 relabeling의 원본 출처는 불명확해 네 묶음이 독립적인 네 경기는 아니다. 현재 모델은 공만 검출하며 득점·바운드·라켓 타격 판정 모델은 아니다.

## 1 Roboflow 데이터 준비

1. [dachui Table Tennis Ball v2](https://universe.roboflow.com/dachui/table-tennis-ball-um5tc-jrseu/dataset/2)에서 **Download → YOLOv11 → Download ZIP**을 선택한다. 내보내기 화면에서 요청하면 Roboflow에 로그인한다. [Roboflow 내보내기 안내](https://docs.roboflow.com/datasets/versions/dataset-versions/exporting-data)
2. ZIP을 풀어 아래 구조로 둔다. `data.yaml`은 ZIP에 들어 있던 원본을 사용한다. v2 공개 페이지의 분할은 학습 888장, 검증 110장, 테스트 114장이다. `test`는 최종 평가용으로 보관한다.

```text
datasets/table-tennis-ball-v2/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 2 RTX 5050 학습 환경 준비

저장소 루트에서 PowerShell 명령을 실행한다. 강의 환경과 같은 Python 3.14를 사용하며, `uv sync`는 이 프로젝트 안의 `.venv`에 의존성을 설치한다. 프로젝트 설정에 PyTorch와 torchvision의 CUDA 13.0 빌드를 지정했다. [PyTorch 설치 안내](https://pytorch.org/get-started/locally/) · [uv PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/)

```powershell
Set-Location '.\04-table-tennis-ai-referee'
uv sync --system-certs
uv run python -X utf8 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(2, device='cuda').sum().item())"
```

GPU 이름과 `2.0`이 나오면 CUDA에서 실제 연산을 실행한 것이다.

## 3 라벨 확인과 학습

```powershell
uv run python -X utf8 train_detector.py --check-only
uv run python -X utf8 train_detector.py --epochs 1
uv run python -X utf8 train_detector.py --epochs 30
```

- `--check-only`는 분할별 이미지와 공 박스 수, 라벨 예시를 출력한다. 강의에서 배운 것처럼 한 줄은 `클래스 번호 중심x 중심y 너비 높이` 순서이고, 좌표는 이미지 크기로 나눈 값이다. [Roboflow YOLO 라벨 형식](https://roboflow.com/formats/yolo)
- 1 epoch는 학습부터 결과 저장까지 연결을 확인하는 실행이다. 확인한 뒤 30 epoch로 기준 모델을 학습한다. `yolo11n.pt`가 폴더에 없으면 첫 실행에서 사전학습 가중치를 내려받는다. GPU 메모리가 부족하면 같은 명령에 `--batch 2`를 붙인다. [공식 가중치](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11n.pt)
- 다른 위치에 압축을 풀었다면 명령에 `--data 'C:\경로\data.yaml'`을 붙인다. 스크립트가 ZIP에 들어 있던 경로를 검사하고, 실행용 YAML을 결과 폴더에 저장한다.

## 4 결과 확인

실행마다 새 `runs/ball-*` 폴더를 만든다. `weights/best.pt`는 검증에 사용한 가중치이고, `results.csv`는 epoch별 학습 결과다. `val_prediction.jpg`에서 검증 이미지 한 장의 공 검출 결과를 볼 수 있다. 콘솔에 나오는 mAP는 이미지에서 공 박스를 얼마나 잘 찾는지 측정한다. 추후 랠리 득점 모델에 사용할 공 위치를 얻는 단계다.

RTX 5050 Laptop GPU에서 임시 이미지로 1 epoch를 실행해 `best.pt`, `results.csv`, `val_prediction.jpg` 생성을 확인했다. Roboflow 데이터로 학습한 검출 성능은 별도로 측정한다.
