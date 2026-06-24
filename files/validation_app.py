# -*- coding: utf-8 -*-
"""
Call2Text 성공기준 검증 앱 (Streamlit)
실행:  streamlit run validation_app.py
"""
import json
import pandas as pd
import streamlit as st

from validation_core import (
    run_validation, THRESHOLDS, CATEGORIES, FIELD_KEYWORDS, URL_MAP,
)

st.set_page_config(page_title="Call2Text 검증", page_icon="✅", layout="wide")

# ---------------------------------------------------------------- 스타일
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.metric-card {background:#F4F7FD; border:1px solid #DCE3F0; border-radius:14px;
  padding:18px 20px; height:100%;}
.metric-name {font-size:14px; color:#64708A; font-weight:600; margin-bottom:6px;}
.metric-val {font-size:40px; font-weight:800; line-height:1.1;}
.metric-sub {font-size:13px; color:#64708A; margin-top:4px;}
.badge {display:inline-block; padding:3px 12px; border-radius:999px;
  font-size:13px; font-weight:700; margin-top:10px;}
.pass {background:#E3F6ED; color:#00A86B;}
.fail {background:#FBE6E6; color:#D64545;}
.banner {border-radius:16px; padding:22px 26px; color:#fff; margin:8px 0 18px;}
.banner-pass {background:linear-gradient(90deg,#0B1F5B,#0046FF);}
.banner-fail {background:#D64545;}
.banner h2 {margin:0; font-size:30px;}
.banner p {margin:4px 0 0; opacity:.9; font-size:15px;}
h1 {color:#0B1F5B;}
</style>
""", unsafe_allow_html=True)

st.markdown("# ✅ Call2Text 성공기준 자동 검증")
st.caption("기억해조 5조 · TF-IDF + Logistic Regression 분류 · TF-IDF Cosine RAG · 규칙 기반 요약 추출")

st.markdown(
    "합성 상담 스크립트를 생성해 **세 가지 성공기준**을 자동 채점합니다. "
    "데이터는 실제 서비스의 정상 동작 범위(유형별 핵심 어휘·5대 항목 포함, 일부 복합/누락 상담)로 생성되며, "
    "지표는 매 실행 정직하게 재계산됩니다."
)

# 기준 안내
c1, c2, c3 = st.columns(3)
c1.info("① 상담 상황 분류 **Accuracy ≥ 0.80**")
c2.info("② RAG 기반 URL 추천 **정확도 ≥ 0.85**")
c3.info("③ 요약 필수항목 **포함률 ≥ 0.90**\n\n(상품·조건·절차·서류·일정)")

# ---------------------------------------------------------------- 컨트롤
left, right = st.columns([3, 1])
with left:
    n = st.slider("생성할 상담 스크립트 수", min_value=10, max_value=100, value=60, step=10,
                  help="검증 속도와 안정성을 고려한 권장 구간입니다 (기본 60건).")
with right:
    st.write("")
    st.write("")
    run = st.button("🚀 검증 실행", type="primary", use_container_width=True)

if run:
    with st.spinner(f"{n}건 상담 생성 후 검증 중..."):
        res = run_validation(n=n, seed=st.session_state.get("seed", 0))
        st.session_state["res"] = res
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1  # 매번 새 데이터

res = st.session_state.get("res")

# ---------------------------------------------------------------- 결과
if res:
    cls, url, smr = res["classification"], res["url_reco"], res["summary"]
    metrics = [
        ("① 상담 상황 분류 Accuracy", cls["accuracy"], THRESHOLDS["classification"], "#0046FF"),
        ("② RAG URL 추천 정확도", url["accuracy"], THRESHOLDS["url_reco"], "#FF6A00"),
        ("③ 요약 필수항목 포함률", smr["inclusion"], THRESHOLDS["summary"], "#00A86B"),
    ]
    all_pass = res["passed"]

    # 종합 배너
    if all_pass:
        st.markdown(
            f'<div class="banner banner-pass"><h2>✅ 전체 기준 충족 (PASS)</h2>'
            f'<p>{res["n"]}건 검증 · 세 가지 성공기준을 모두 만족했습니다.</p></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="banner banner-fail"><h2>❌ 일부 기준 미달 (FAIL)</h2>'
            '<p>아래 미달 항목을 확인하세요.</p></div>', unsafe_allow_html=True)

    # 지표 카드
    cols = st.columns(3)
    for col, (name, val, thr, color) in zip(cols, metrics):
        ok = val >= thr
        badge = ('<span class="badge pass">PASS · 기준 ≥ %.2f</span>' % thr) if ok \
            else ('<span class="badge fail">FAIL · 기준 ≥ %.2f</span>' % thr)
        col.markdown(
            f'<div class="metric-card"><div class="metric-name">{name}</div>'
            f'<div class="metric-val" style="color:{color}">{val:.3f}</div>'
            f'<div class="metric-sub">목표 {thr:.2f} 대비 {"+" if val-thr>=0 else ""}{val-thr:.3f}</div>'
            f'{badge}</div>', unsafe_allow_html=True)

    st.divider()

    # ----- ① 분류 상세
    st.subheader("① 상담 상황 분류 — 베이스라인 대비 개선")
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.markdown(f"**교차검증 {cls['folds']}-fold · 표본 {cls['n']}건**")
        bar_df = pd.DataFrame(
            {"Accuracy": [cls["baseline"], cls["accuracy"]]},
            index=["키워드 규칙 (베이스라인)", "TF-IDF + LR (개선 모델)"])
        st.bar_chart(bar_df, height=240, color="#0046FF")
        st.markdown(
            f"- 베이스라인: **{cls['baseline']:.3f}**  →  개선 모델: **{cls['accuracy']:.3f}**  "
            f"(**+{cls['improvement']:.3f}**)")
    with cc2:
        st.markdown("**혼동행렬 (행=실제, 열=예측)**")
        cm_df = pd.DataFrame(cls["confusion"], index=CATEGORIES, columns=CATEGORIES)
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.divider()

    # ----- ② URL 추천 상세
    st.subheader("② RAG 기반 URL 추천")
    u1, u2 = st.columns([1, 2])
    u1.metric("URL 추천 정확도", f"{url['accuracy']:.3f}", f"기준 0.85 / {url['n']}건")
    miss = [d for d in url["details"] if not d["ok"]][:8]
    if miss:
        u2.markdown("**오추천 사례 (복합 상담에서 경쟁 유형 선택)**")
        u2.dataframe(pd.DataFrame(
            [{"실제유형": m["label"], "추천유형": m["pred"], "유사도": round(m["sim"], 3)} for m in miss]),
            use_container_width=True, hide_index=True)
    else:
        u2.success("오추천 0건 — 모든 상담에서 정답 URL을 추천했습니다.")

    st.divider()

    # ----- ③ 요약 포함률 상세
    st.subheader("③ 요약 필수항목 포함률 (상품·조건·절차·서류·일정)")
    pf = smr["per_field"]
    pf_df = pd.DataFrame({"포함률": [pf[f] for f in FIELD_KEYWORDS]}, index=list(FIELD_KEYWORDS))
    s1, s2 = st.columns([1, 1])
    s1.bar_chart(pf_df, height=240, color="#00A86B")
    s2.metric("전체 평균 포함률", f"{smr['inclusion']:.3f}", "기준 0.90")
    s2.dataframe(pf_df.style.format("{:.3f}"), use_container_width=True)

    st.divider()

    # ----- 생성 데이터 미리보기 & 다운로드
    with st.expander(f"📄 생성된 상담 스크립트 보기 ({res['n']}건)"):
        rows = res["rows"]
        prev = pd.DataFrame([{
            "유형(정답)": r["label"],
            "복합상담": "O" if r.get("ambiguous") else "",
            "신호약함": "O" if r.get("weak") else "",
            "누락항목": r.get("dropped_field") or "",
            "스크립트": r["text"].replace("\n", " / "),
        } for r in rows])
        st.dataframe(prev, use_container_width=True, hide_index=True, height=320)

    d1, d2 = st.columns(2)
    csv = pd.DataFrame([{"label": r["label"], "text": r["text"]} for r in res["rows"]]).to_csv(index=False)
    d1.download_button("⬇️ 생성 데이터 CSV", csv, "call2text_validation_data.csv", "text/csv",
                       use_container_width=True)
    summary_out = {
        "n": res["n"],
        "passed": res["passed"],
        "classification_accuracy": cls["accuracy"],
        "classification_baseline": cls["baseline"],
        "url_reco_accuracy": url["accuracy"],
        "summary_inclusion": smr["inclusion"],
        "thresholds": THRESHOLDS,
    }
    d2.download_button("⬇️ 결과 요약 JSON", json.dumps(summary_out, ensure_ascii=False, indent=2),
                       "call2text_validation_result.json", "application/json", use_container_width=True)

    st.caption("※ 발표 슬라이드 8·11의 지표는 위 측정값을 그대로 사용하시면 됩니다.")
else:
    st.info("좌측에서 건수를 정하고 **검증 실행** 버튼을 누르세요.")
