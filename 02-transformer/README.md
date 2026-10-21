# 02 · Transformer 학습

영어 문장을 독일어로 번역하는 Encoder–Decoder Transformer를 그림과 실제 코드로 살펴봅니다.

## 원본과 실습본

| 폴더 | 역할 |
| --- | --- |
| `reference/` | 기존 `transformer` 폴더의 구조를 보존한 참고용 코드. 직접 만든 ELI5 교재도 포함합니다. |
| `training/` | 기존 `success_transformer` 폴더의 실습본. `pyproject.toml`, `uv.lock`, Python 버전 파일, Multi30k 다운로드 스크립트를 포함합니다. |

두 폴더는 학습 비교용으로 따로 보관합니다. 이름이 `training`이라는 것은 실습 환경 구성본이라는 뜻이며, 학습 완료나 번역 성능 검증을 뜻하지 않습니다.

## 그림과 코드 교재

브라우저에서 `reference/docs/book/index.html`을 열면 다음 페이지를 오갈 수 있습니다.

- 프로젝트 개요와 사전 지식
- 문장 한 쌍의 데이터 흐름
- Encoder 원본 그림의 9개 위치와 실제 코드 대조
- 60개 정의와 4개 실행 구간의 단계별 해설
- 전체 Python 소스 원문

교재 페이지: [전체 목차](reference/docs/book/index.html) · [그림별 Encoder 해설](reference/docs/book/Transformer_Encoder_Blocks_ELI5.html) · [함수 해설](reference/docs/book/Transformer_Code_Walkthrough_ELI5.html)

GitHub의 파일 보기에서는 HTML이 페이지로 실행되지 않습니다. 로컬 브라우저로 열거나 별도의 정적 웹 호스팅을 사용하세요.

교재 내용은 `reference` 코드 기준의 스냅샷입니다. 코드를 수정한 후에는 다음 명령으로 다시 생성하고 검사할 수 있습니다.

```powershell
cd 02-transformer/reference
python docs/book/build_book.py
python docs/book/check_book.py
```

`03-web-visualizations/transformer-eli5-book/dist/`는 이 교재의 웹 게시용 복사본입니다. 교재를 수정했다면 해당 복사본도 갱신해야 합니다.

## 실습 실행

[training/SETUP.md](training/SETUP.md)를 확인하세요. 원본의 README와 저자 표기도 각 폴더에 보존했습니다.

## 현재 구현의 범위

학습·검증·가중치 저장 코드가 있습니다. 평가도 정답 앞부분을 제공하는 방식이며, 새 문장을 시작 토큰부터 끝까지 생성하는 별도 번역 루프는 없습니다. 기존 BLEU 호출에는 입력 자료 구조의 불일치가 있어 기록값을 표준 문장 BLEU로 해석하지 않도록 교재에 설명했습니다.

[전체 목차로 돌아가기](../README.md)
