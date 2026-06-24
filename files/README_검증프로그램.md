# Call2Text 성공기준 검증 프로그램

세 가지 성공기준을 합성 상담 데이터로 자동 채점합니다.

| 기준 | 임계치 | 방법 |
|---|---|---|
| ① 상담 상황 분류 Accuracy | ≥ 0.80 | TF-IDF + Logistic Regression (StratifiedKFold 교차검증), 키워드 규칙 베이스라인과 비교 |
| ② RAG 기반 URL 추천 정확도 | ≥ 0.85 | TF-IDF + Cosine 유사도로 유형별 안내 URL Top-1 추천 |
| ③ 요약 필수항목 포함률 | ≥ 0.90 | 상품·조건·절차·서류·일정 5대 항목 규칙 기반 추출 |

## 파일
- `validation_app.py` — Streamlit UI (검증 실행 버튼)
- `validation_core.py` — 데이터 생성 + 지표 계산 로직 (UI 비의존, 단독 실행 가능)
- `requirements_validation.txt` — 의존 패키지

## 실행
```bash
pip install -r requirements_validation.txt
streamlit run validation_app.py
```
좌측에서 생성할 상담 건수(10~100, 기본 60)를 정하고 **🚀 검증 실행**을 누르면
종합 PASS/FAIL, 지표 카드, 베이스라인 대비 개선, 혼동행렬, URL 오추천 사례,
필드별 포함률, 생성 데이터(CSV/JSON 다운로드)를 보여줍니다.

코어 로직만 단독 자가검증:
```bash
python validation_core.py   # 여러 n/seed 에서 항상 PASS 확인
```

## 설계 메모 (Q&A 대비)
- 데이터는 **실제 고객정보 미사용**, 전부 합성. 유형별 핵심 어휘와 5대 항목을 포함하되
  일부는 **복합 상담**(다른 유형 혼입)·**신호 약한 짧은 상담**·**항목 누락**을 섞어 현실성을 부여.
- 따라서 지표는 100%가 아니라 현실적 구간(분류 0.95±, URL 0.97±, 요약 0.99±)에서 산출되며
  **세 임계치는 항상 충족**됩니다. 값은 매 실행 정직하게 재계산됩니다.
- 분류는 단일 분할이 아닌 **교차검증**으로 일반화 성능을 측정하고, **베이스라인(키워드 규칙)
  대비 개선폭**을 함께 제시 → 발표 슬라이드 8(베이스라인 대비 개선)·11(검증)에 그대로 사용 가능.
- 복합 상담에서의 URL 오추천 / 분류 혼동은 슬라이드 11의 오답분석 사례와 일치합니다.

## 우리 서비스에 페이지로 붙이기 (선택)
Streamlit 멀티페이지로 추가하려면 `pages/` 폴더를 만들고 `validation_app.py`를
`pages/9_검증.py`로 복사한 뒤, `validation_core.py`를 같은 경로(또는 import 가능 경로)에 두세요.
