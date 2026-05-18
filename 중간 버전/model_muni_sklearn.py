#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import joblib
import warnings

warnings.filterwarnings('ignore')

# =====================
# 1. 경로 설정 및 파일 로드 (문의 전용)
# =====================
file_path = "대청호_문의_순수데이터.csv" 
save_dir = "Model_7Days_Muni"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"📁 새 폴더를 생성했습니다: {save_dir}")

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"✅ [문의] 순수 데이터 로드 완료! 총 {len(df)}행")
except Exception as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    exit()

# =====================
# 2. 7일 선행 시계열 데이터 매칭 (Time-Shift)
# =====================
print("⏳ '오늘의 환경'과 '7일 뒤의 조류 세포수'를 짝짓는 중입니다...")

df['조사일'] = pd.to_datetime(df['조사일'])
df = df.sort_values('조사일')

# 🚨 업데이트: 컨닝(과거 세포수) 삭제! 순수 환경 인자 & 새로운 파생변수 추가!
columns_X = [
    '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
    '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
    '수위(EL.m)', '저수량(백만㎥)', '유입량(㎥/s)', '총방류량(㎥/s)', 
    'TN(㎎/L)', 'TP(㎎/L)', 'NP_Ratio', 'Retention_Index'
]

# 결측치가 있는 행은 학습할 수 없으므로 제거
df_clean = df.dropna(subset=columns_X + ['조사일']).copy()

df_target = df[['조사일', '유해남조류 세포수 (cells/㎖)']].copy()
df_target = df_target.dropna() # 정답 없는 날 제거
df_target = df_target.rename(columns={'조사일': '타겟_조사일', '유해남조류 세포수 (cells/㎖)': 'Target_7일뒤_세포수'})

# 오늘 날짜에 7일을 더해서 '예측목표일' 생성
df_clean['예측목표일'] = df_clean['조사일'] + pd.Timedelta(days=7)

df_clean = df_clean.sort_values('예측목표일')
df_target = df_target.sort_values('타겟_조사일')

# 💡 순수 데이터의 '불규칙한 간격'을 해결하는 핵심 로직!
# 예측목표일과 가장 가까운(최대 2일 오차 허용) 실제 측정일을 찾아서 붙여줌
df_merged = pd.merge_asof(
    df_clean, 
    df_target,
    left_on='예측목표일',
    right_on='타겟_조사일',
    direction='nearest',
    tolerance=pd.Timedelta(days=2)
)

df_final = df_merged.dropna(subset=['Target_7일뒤_세포수']).copy()
print(f"✅ 7일 선행 매칭 완료! 최종 학습 가능 데이터: {len(df_final)}행")

# =====================
# 3. 학습 변수 분리 및 스케일링
# =====================
X = df_final[columns_X].values
y = df_final['Target_7일뒤_세포수'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

n_clusters = 3  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = pd.DataFrame(X_scaled, columns=columns_X)  
df_clustered['y'] = y_scaled.flatten()
df_clustered['cluster'] = clusters

# =====================
# 4. 하이퍼파라미터 탐색 및 학습 (Scikit-Learn MLP)
# =====================
hidden_layer_options = [3, 4]
input_size_options = [64, 128]
activation_options = ['relu', 'tanh']
epoch_options = [400, 600]
train_test_split_options = [0.8] # 테스트용으로 하나만 고정

results = []
best_r2 = -float('inf')
best_model = None

print("🚀 [문의] 모델 딥러닝 학습 시작...")

for hl, isize, act_name, ep, tr in itertools.product(
    hidden_layer_options, input_size_options, activation_options, epoch_options, train_test_split_options):

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for c in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster']==c]
        if len(cluster_data) < 2: continue 
        
        n_train = int(len(cluster_data) * tr)
        # 동기의 랜덤 스플릿(무작위 섞기) 로직 유지 
        cluster_train = cluster_data.sample(n=n_train, random_state=42)
        cluster_test = cluster_data.drop(cluster_train.index)
        
        X_train_list.append(cluster_train.drop(columns=['y','cluster']).values)
        X_test_list.append(cluster_test.drop(columns=['y','cluster']).values)
        y_train_list.append(cluster_train['y'].values.ravel())
        y_test_list.append(cluster_test['y'].values.ravel())
    
    X_train, X_test = np.vstack(X_train_list), np.vstack(X_test_list)
    y_train, y_test = np.concatenate(y_train_list), np.concatenate(y_test_list)

    hidden_layer_sizes = tuple([isize] * hl)
    act_fn = 'relu' if act_name == 'relu' else 'tanh'
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=act_fn,
        max_iter=ep,
        learning_rate_init=0.01,
        early_stopping=True, 
        random_state=42
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # 마이너스 예측값은 0으로 보정
    y_pred_orig = np.maximum(0, y_pred_orig)
    
    r2 = r2_score(y_orig, y_pred_orig)
    mae = mean_absolute_error(y_orig, y_pred_orig)

    results.append({
        'hidden_layers': hl, 'input_size': isize, 'activation': act_name,
        'epochs': ep, 'train_ratio': tr, 'test_mae': mae, 'test_r2': r2
    })

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
    
    print(f"[+] R2: {r2:.4f} | HL:{hl}, IS:{isize}, ACT:{act_name}, EP:{ep}")

# =====================
# 5. 저장
# =====================
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(save_dir, "Result_Muni.csv"), index=False, encoding='utf-8-sig')

joblib.dump(best_model, os.path.join(save_dir, "best_muni_mlp.pkl"))
joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X_muni.pkl"))
joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y_muni.pkl"))
joblib.dump(kmeans, os.path.join(save_dir, "kmeans_muni.pkl"))

print(f"\n✨ [문의] 모델 완료! 폴더: {save_dir}")