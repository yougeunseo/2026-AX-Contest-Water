#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# [경로 에러 해결] 파이썬 파일이 있는 곳을 강제로 기준 폴더로 설정
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


# 폰트 설정 (윈도우 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("📊 조류경보 예측 적중률 계산 및 시계열 시각화 차트 생성을 시작합니다...\n")

input_file_path = "대청호_통합_순수데이터.csv"
base_dir = "Model_Integrated_Result"

# 데이터 로드
try:
    try:
        df_master = pd.read_csv(input_file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_master = pd.read_csv(input_file_path, encoding='cp949')
    df_master['조사일'] = pd.to_datetime(df_master['조사일'])
except Exception as e:
    print(f"❌ 통합 데이터를 읽지 못했습니다: {e}")
    exit()

def get_alert_level(cells):
    if cells >= 1000000: return '조류대발생'
    elif cells >= 10000: return '경계'
    elif cells >= 1000: return '관심'
    else: return '해제'

regions = ['문의', '추동', '회남']

for region in regions:
    print(f"========================================")
    print(f"📍 [{region}] 지점 평가 및 시각화 중...")

    model_dir = os.path.join(base_dir, region)
    df_region = df_master[df_master['채수위치'] == region].copy().sort_values('조사일')
    
    if len(df_region) == 0: continue

    # =====================
    # 1. 7일 선행 정답지(Ground Truth) 매칭
    # =====================
    if region == '문의':
        columns_X_final = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            'TN(㎎/L)', 'TP(㎎/L)', 'NP_Ratio', '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 'Retention_Index',
            '유해남조류 세포수 (cells/㎖)'
        ]
    else:
        columns_X_final = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 'Retention_Index',
            '유해남조류 세포수 (cells/㎖)'
        ]

    df_clean = df_region.dropna(subset=columns_X_final + ['조사일']).copy()
    df_target = df_region[['조사일', '유해남조류 세포수 (cells/㎖)']].copy().dropna()
    df_target = df_target.rename(columns={'조사일': '타겟_조사일', '유해남조류 세포수 (cells/㎖)': 'Target_7일뒤_세포수'})

    df_clean['예측목표일'] = df_clean['조사일'] + pd.Timedelta(days=7)
    df_clean = df_clean.sort_values('예측목표일')
    df_target = df_target.sort_values('타겟_조사일')

    df_merged = pd.merge_asof(
        df_clean, df_target,
        left_on='예측목표일', right_on='타겟_조사일',
        direction='nearest', tolerance=pd.Timedelta(days=2)
    )
    df_final = df_merged.dropna(subset=['Target_7일뒤_세포수']).copy()

    # =====================
    # 2. AI 예측 수행
    # =====================
    try:
        scaler_X = joblib.load(os.path.join(model_dir, "scaler_X_7days.pkl"))
        scaler_y = joblib.load(os.path.join(model_dir, "scaler_y_7days.pkl"))
        model = joblib.load(os.path.join(model_dir, "best_7days_model.pkl"))
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}\n")
        continue

    X = df_final[columns_X_final].values
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled)
    
    if len(y_pred_scaled.shape) == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_pred_final = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_pred_final = np.maximum(0, y_pred_final)

    # =====================
    # 3. 조류경보제 적중률(Accuracy) 계산
    # =====================
    df_final['Actual_Level'] = df_final['Target_7일뒤_세포수'].apply(get_alert_level)
    df_final['Pred_Level'] = pd.Series(y_pred_final).apply(get_alert_level)

    # 실제 경보 등급과 AI 예측 등급이 일치하는 비율 계산
    accuracy = (df_final['Actual_Level'] == df_final['Pred_Level']).mean() * 100
    print(f"🎯 [{region}] 조류경보 발령 등급 예측 적중률: {accuracy:.2f}%")

    # =====================
    # 4. 시계열 꺾은선 차트 그리기 (보고서 제출용)
    # =====================
    plt.figure(figsize=(12, 6))
    
    # 실제값(파란선) vs 예측값(주황 점선)
    plt.plot(df_final['타겟_조사일'], df_final['Target_7일뒤_세포수'], label='실제 측정값', color='dodgerblue', marker='o', markersize=4, linewidth=1.5)
    plt.plot(df_final['타겟_조사일'], y_pred_final, label='AI 7일 선행 예측값', color='darkorange', linestyle='--', marker='x', markersize=4, linewidth=1.5)

    # 조류경보제 가이드라인 (가로선)
    plt.axhline(y=1000, color='gold', linestyle='-', linewidth=2, alpha=0.8, label='관심 단계 (1,000 cells/mL)')
    plt.axhline(y=10000, color='red', linestyle='-', linewidth=2, alpha=0.6, label='경계 단계 (10,000 cells/mL)')

    plt.title(f'[{region}] 대청호 조류 대발생 7일 선행 예측 시계열 (경보 적중률: {accuracy:.1f}%)', fontsize=16, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('유해남조류 세포수 (cells/㎖) - 로그 스케일', fontsize=12)
    
    # 남조류는 0에서 수십만까지 폭증하므로 보기 좋게 y축을 로그 스케일로 압축
    plt.yscale('symlog', linthresh=1000) 
    
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(model_dir, f"{region}_시계열예측_그래프.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"📈 [{region}] 시각화 차트 저장 완료 ➡️ {plot_path}\n")

print("✨ 보고서에 넣을 [적중률]과 [시계열 차트] 생성이 모두 완료되었습니다! 고생하셨습니다!")