import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 웹페이지 기본 설정
st.set_page_config(page_title="대청호 조류 조기경보 시스템", layout="wide")

st.title("🌊 대청호 조류 대발생 조기경보 AI 대시보드")
st.markdown("오늘 측정한 수질 데이터를 입력하면, AI 모델이 **7일 뒤**의 조류경보 발령 단계를 예측합니다.")

# ==========================================
# 1. 저장된 AI 뇌(.pkl) 불러오기
# ==========================================
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('final_rf_model.pkl')
        sc = joblib.load('scaler.pkl')
        km = joblib.load('kmeans.pkl')
        return rf, sc, km
    except:
        return None, None, None

rf_model, scaler, kmeans = load_models()

if rf_model is None:
    st.error("🚨 AI 모델 파일을 찾을 수 없어! 같은 폴더에 .pkl 파일 3개가 있는지 확인해 줘.")
    st.stop()

# ==========================================
# 2. 사용자 입력 창 (UI)
# ==========================================
region = st.selectbox("📍 예측할 채수위치를 선택하세요", ["문의", "추동", "회남"])

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("💧 1~5순위 환경 인자")
    transparency = st.number_input("투명도 (m)", value=1.5, step=0.1)
    do_level = st.number_input("DO 용존산소 (㎎/L)", value=9.0, step=0.1)
    temp = st.number_input("수온 (℃)", value=25.0, step=0.1)
    rainfall = st.number_input("일강수량 (mm)", value=0.0, step=1.0)
    chla = st.number_input("Chl-a (㎎/㎥)", value=15.0, step=1.0)

with col2:
    st.subheader("📊 6~10순위 환경 인자")
    tp = st.number_input("TP (㎎/L)", value=0.05, step=0.01)
    tn = st.number_input("TN (㎎/L)", value=2.0, step=0.1)
    np_ratio = st.number_input("N/P 비율", value=40.0, step=1.0)
    retention = st.number_input("체류시간 지표", value=50.0, step=1.0)
    discharge = st.number_input("총방류량 (㎥/s)", value=10.0, step=1.0)

st.divider()

# ==========================================
# 3. 진짜 AI 예측 로직 실행
# ==========================================
if st.button("🚀 7일 뒤 AI 예측 돌리기", use_container_width=True):
    
    # 지역명을 0과 1로 변환
    loc_muni = 1 if region == "문의" else 0
    loc_chudong = 1 if region == "추동" else 0
    loc_hoenam = 1 if region == "회남" else 0
    
    # 입력된 값을 하나의 배열로 묶기 (주의: 학습할 때 쓰인 CSV의 컬럼 순서와 일치해야 함!)
    # 일반적인 순서: 수온, DO, 투명도, 강수량, Chl-a, TP, TN, NP비율, 체류시간, 총방류량, 문의, 추동, 회남
    input_array = np.array([[
        temp, do_level, transparency, rainfall, chla, 
        tp, tn, np_ratio, retention, discharge, 
        loc_muni, loc_chudong, loc_hoenam
    ]])

    try:
        # STEP 1: 스케일링 (수치 정규화)
        scaled_data = scaler.transform(input_array)
        
        # STEP 2: K-Means 클러스터링 패턴 예측
        cluster_num = kmeans.predict(scaled_data).reshape(-1, 1)
        
        # STEP 3: 기존 데이터 + 클러스터 번호 결합 (하이브리드)
        final_input = np.hstack((scaled_data, cluster_num))
        
        # STEP 4: 랜덤 포레스트 최종 예측!
        pred_cells = rf_model.predict(final_input)[0]
        pred_cells = max(0, pred_cells) # 음수 방지

        # 결과 출력
        st.markdown(f"## 🎯 7일 뒤 유해남조류 예측치: **{pred_cells:,.0f}** cells/mL")
        
        if pred_cells >= 1000000:
            st.error("🆘 **[조류 대발생 발령 예상]** 수문 즉시 개방 및 비상 대응이 필요합니다!")
        elif pred_cells >= 10000:
            st.error("🚨 **[경계 단계 발령 예상]** 정수 처리 강화 및 수문 방류를 검토하세요.")
        elif pred_cells >= 1000:
            st.warning("⚠️ **[관심 단계 발령 예상]** 수질 모니터링 주기를 단축하세요.")
        else:
            st.success("✅ **[평상시]** 수질이 안정적일 것으로 예상됩니다.")
            
    except Exception as e:
        st.error(f"⚠️ 예측 중 에러가 발생했어!: {e}")
        st.info("💡 팁: 입력 변수의 갯수나 순서가 학습했던 원본 데이터(csv)와 살짝 다를 때 나는 에러일 수 있어!")