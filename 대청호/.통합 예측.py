#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib

print("🚀 대청호 통합 순수데이터 기반 3개 수역 개별 파일 AI 예측을 시작합니다...\n")

# =====================
# 1. 통합 데이터 로드
# =====================
input_file_path = "대청호_통합_순수데이터.csv"
base_dir = "Model_Integrated_Result" # AI 모델(.pkl)들이 들어있는 최상위 폴더 경로

try:
    try:
        df_master = pd.read_csv(input_file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_master = pd.read_csv(input_file_path, encoding='cp949')
    print(f"✅ 마스터 통합 데이터 로드 완료! (총 {len(df_master)}행)\n")
except Exception as e:
    print(f"❌ 통합 데이터를 읽지 못했습니다: {e}")
    exit()

regions = ['문의', '추동', '회남']

# 🌟 [추가됨] 조류경보제 발령 기준 판별 함수
def get_alert_level(cells):
    if cells >= 1000000:
        return '조류대발생'
    elif cells >= 10000:
        return '경계'
    elif cells >= 1000:
        return '관심'
    else:
        return '해제'

# =====================
# 2. 지역별 순회하며 예측 및 '개별 파일'로 저장
# =====================
for region in regions:
    print(f"========================================")
    print(f"📍 [{region} 지점] AI 뇌(.pkl) 연결 및 예측 중...")
    
    model_dir = os.path.join(base_dir, region)
    
    df_new = df_master[df_master['채수위치'] == region].copy()
    
    if len(df_new) == 0:
        print(f"⚠️ '{region}' 지점의 데이터가 없어 건너뜁니다.\n")
        continue

    # AI 뇌 및 변환 도구 불러오기
    try:
        scaler_X = joblib.load(os.path.join(model_dir, "scaler_X_7days.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, "scaler_y_7days.pkl"))
        model = joblib.load(os.path.join(model_dir, "best_7days_model.pkl"))
    except Exception as e:
        print(f"❌ [{region}] 모델 파일을 찾을 수 없습니다: {e}\n")
        continue

    # AI 두뇌 피처 개수 동기화
    if region == '문의':
        columns_X_final = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            'TN(㎎/L)', 'TP(㎎/L)', 'NP_Ratio', '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 'Retention_Index',
            '유해남조류 세포수 (cells/㎖)'
        ]
    else: # 추동, 회남
        columns_X_final = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 'Retention_Index',
            '유해남조류 세포수 (cells/㎖)'
        ]

    if df_new[columns_X_final].isnull().values.any():
        df_new = df_new.dropna(subset=columns_X_final)

    X_new = df_new[columns_X_final].values
    if len(X_new) == 0:
        print(f"⚠️ [{region}] 예측할 유효한 데이터가 없습니다.\n")
        continue

    # 스케일링 및 AI 예측
    X_new_scaled = scaler_X.transform(X_new)
    y_pred_scaled = model.predict(X_new_scaled)
    
    if len(y_pred_scaled.shape) == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    # 역변환
    y_pred_final = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_pred_final = np.maximum(0, y_pred_final) 

    # 🌟 [추가됨] 예측된 세포수를 기반으로 경보 단계 자동 맵핑
    df_new['7일뒤_AI예측_세포수'] = np.round(y_pred_final, 0)
    df_new['조류경보제 발령 예상'] = df_new['7일뒤_AI예측_세포수'].apply(get_alert_level)
    
    output_filename = f"대청호_{region}_예측결과.csv"
    df_new.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"🎯 [{region}] 예측 완료! 개별 파일 생성됨 ➡️ {output_filename}\n")

print("✨ 모든 지점의 개별 예측 결과 파일(조류경보제 등급 포함) 작성이 완료되었습니다!")