# 03 · 웹 학습 자료

기존 `sites` 폴더의 HTML·CSS·JavaScript 산출물을 보관합니다. 별도의 빌드나 서버 프로그램 없이 읽는 정적 페이지입니다.

| 폴더 | 내용 |
| --- | --- |
| `vector-angle/dist/` | 벡터·각도 학습 페이지 |
| `transformer-eli5-book/dist/` | Transformer 교재 7개 페이지와 그림의 웹 게시용 복사본 |

## 로컬에서 보기

각 `dist/index.html`을 브라우저로 열면 됩니다. 로컬 HTTP 서버가 필요하다면 저장소 루트에서 다음과 같이 실행할 수도 있습니다.

```powershell
python -m http.server 8000 --directory 03-web-visualizations
```

- 벡터·각도: `http://localhost:8000/vector-angle/dist/`
- Transformer 교재: `http://localhost:8000/transformer-eli5-book/dist/`

## 교재 수정과 게시

Transformer 교재의 원본과 생성·검사 스크립트는 [02-transformer/reference/docs/book](../02-transformer/reference/docs/book/)에 있습니다. 원본 교재를 수정한 후 HTML·CSS·JavaScript와 `assets/`를 게시용 `dist/`에 복사하세요.

이 폴더를 GitHub에 저장하는 것과 웹사이트를 공개 게시하는 것은 별개의 작업입니다. GitHub 저장소의 HTML 파일 보기만으로 실행 화면이 제공되지는 않습니다. ZIP과 배포 서비스의 로그인·인증 정보는 보관하지 않습니다.

[전체 목차로 돌아가기](../README.md)
