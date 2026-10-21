# SeSAC · 새싹 학습 기록

새싹 과정에서 작성하거나 살펴본 코드, 실습 환경, 학습 교재를 주제별로 정리합니다. Codyssey 저장소처럼 각 학습 폴더의 README에서 목적과 실행 방법을 확인할 수 있습니다.

## 학습 목차

| 폴더 | 내용 | 바로가기 |
| --- | --- | --- |
| `01-python-basics/` | Python 시작 코드와 노트북 | [학습 안내](01-python-basics/README.md) |
| `02-transformer/reference/` | 살펴본 Transformer 원본 코드와 ELI5 교재 | [Transformer 학습 안내](02-transformer/README.md) |
| `02-transformer/training/` | uv 환경과 Multi30k 준비 스크립트가 포함된 실습본 | [실행 안내](02-transformer/training/SETUP.md) |
| `03-web-visualizations/` | 벡터·각도 학습 페이지와 Transformer 교재의 웹 게시용 파일 | [페이지 안내](03-web-visualizations/README.md) |

```text
sesac/
├── 01-python-basics/
│   ├── main.py
│   └── sesac.ipynb
├── 02-transformer/
│   ├── reference/              # 원본 구조를 보존한 학습용 코드
│   │   └── docs/book/          # 그림·코드·함수 해설 7개 페이지
│   └── training/               # Python 3.9 · uv · 데이터 준비 스크립트
└── 03-web-visualizations/
    ├── vector-angle/dist/
    └── transformer-eli5-book/dist/
```

## 읽는 순서

1. [Python 기초](01-python-basics/README.md)를 확인합니다.
2. [Transformer 학습 안내](02-transformer/README.md)에서 원본과 실습본의 역할을 구분합니다.
3. `02-transformer/reference/docs/book/index.html`을 브라우저로 열어 그림과 코드를 비교합니다.
4. [실행 안내](02-transformer/training/SETUP.md)에 따라 환경과 데이터를 준비합니다.

## 자료를 추가하는 방법

기존 주제에 이어지는 실습은 해당 폴더 아래에 추가합니다. 새로운 주제는 `04-주제명/`처럼 다음 번호로 만들고, 그 폴더의 README에 목적·관련 파일·실행 방법·확인한 결과를 적은 다음 이 목차에 연결합니다.

각 프로젝트 폴더를 작업 위치로 사용합니다. 의존성이나 가상환경을 저장소 전체에 하나로 합치지 않습니다.

## 보관 범위

- `sesac3` → `01-python-basics`
- `transformer` → `02-transformer/reference`
- `success_transformer` → `02-transformer/training`
- `sites` → `03-web-visualizations`

기존 작업 폴더에서 파일을 복사해 정리했습니다. 원래 폴더와 Git 이력은 변경하지 않았으며 이 저장소는 독립된 하나의 Git 저장소입니다. 하위 프로젝트를 submodule로 등록하지 않습니다.

가상환경, 다운로드 데이터, 모델 가중치, 캐시, 편집기 설정, 배포용 ZIP은 저장하지 않습니다. 기존 예제에 포함된 작은 학습 그래프와 기록 파일은 원본 설명을 읽는 데 필요해 보존합니다. `dist/`는 실제 HTML 산출물이므로 함께 저장합니다.

## 출처

Transformer 참고 코드의 원저자는 Hyunwoong Ko이며 원본 저장소는 [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)입니다. 원문의 저자 표기와 Apache License 2.0 안내를 보존했습니다. 자세한 내용은 [출처 안내](THIRD_PARTY_NOTICES.md)를 확인하세요.
