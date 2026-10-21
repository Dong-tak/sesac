# Transformer 실습본 실행 안내

기존 `success_transformer`의 코드와 uv 설정을 보존한 실습본입니다. 저장소에 데이터와 가상환경은 포함하지 않습니다.

## 준비

저장소 루트에서 PowerShell로 실행합니다. uv가 설치되어 있어야 하며, 환경 구성과 자료 다운로드에는 인터넷이 필요합니다.

```powershell
cd 02-transformer/training
uv sync
uv run python get_multi30k.py
New-Item -ItemType Directory -Path saved,result -Force
```

- `.python-version`: Python 3.9
- `pyproject.toml`과 `uv.lock`: torch 1.10.2, torchtext 0.11.2와 spaCy 및 언어 모델 등의 환경 기록
- `get_multi30k.py`: Multi30k 영어·독일어 자료를 `.data/multi30k/`에 준비

다른 프로젝트의 가상환경을 활성화한 터미널에서는 uv의 `VIRTUAL_ENV` 경고가 나올 수 있습니다. 해당 환경을 `deactivate`하거나 새 터미널에서 시작하세요. 다른 프로젝트의 환경을 사용하기 위해 `--active`를 붙이지 않습니다.

## 학습

```powershell
uv run python train.py
```

현재 `conf.py`의 기본 설정은 1000 epoch, 배치 크기 128, Encoder·Decoder 각각 6층입니다. 짧게 동작을 확인하려면 실행 전에 `epoch = 1`로 바꾸고 장치 메모리에 맞게 `batch_size`를 줄이세요. 이 정리 작업에서는 설정값을 바꾸거나 학습을 실행하지 않았습니다.

모델의 실제 계산 장치는 다음 명령으로 확인할 수 있습니다. 설치한 PyTorch 버전·배포판과 GPU에 따라 CUDA 사용 가능 여부가 달라집니다.

```powershell
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 결과

- `saved/model-검증loss.pt`: 한 epoch의 학습·검증이 끝난 뒤, 이전보다 좋은 검증 loss일 때 저장한 가중치
- `result/train_loss.txt`: 훈련 loss 기록
- `result/test_loss.txt`: 파일 이름과 달리 실제로는 검증 loss 기록
- `result/bleu.txt`: 기존 BLEU 구현의 기록값

`KeyboardInterrupt`는 실행 중단을 뜻하며 학습 완료를 뜻하지 않습니다. 첫 저장 전에 중단하면 그 실행의 가중치가 저장되지 않습니다. 기존 `saved/transformer-base/`의 작은 그래프·텍스트는 참고 코드에 포함되어 있던 예제 자료이며 이번에 새로 학습한 결과가 아닙니다.

가중치만으로 즉시 번역 문장을 만들 수는 없습니다. 실제 번역에는 동일한 모델 설정과 어휘표, 저장한 가중치를 불러오는 코드, 예측 토큰을 다음 입력에 이어 붙이는 생성 루프가 추가로 필요합니다.

[Transformer 학습 안내로 돌아가기](../README.md)
