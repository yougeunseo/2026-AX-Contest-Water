#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import joblib
import matplotlib.pyplot as plt

# 한글 깨짐 방지 폰트 설정 (윈도우)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 환경 설정 및 데이터 로드
# =====================
file_path = r"C:\대청호\대청호_통합_순수데이터.csv"
base_save_dir = r"C:\대청호"

try:
    try:
        df_master = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_master = pd.read_csv(file_path, encoding='cp949')
    print(f"✅ 마스터 데이터 로드 완료! 총 {len(df_master)}행")
except Exception as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    exit()

# 예측 대상 지점 정의
stations = ['문의', '추동', '회남']

# =====================
# 2. 지점별 개별 모델 학습 루프
# =====================
for station in stations:
    print(f"\n========================================")
    print(f"📍 [{station}] 지점 AI 모델 학습 시작")
    print(f"========================================")
    
    # 지점별 전용 폴더 생성 (예: C:\대청호\문의)
    station_dir = os.path.join(base_save_dir, station)
    if not os.path.exists(station_dir):
        os.makedirs(station_dir)
        
    # 해당 지점 데이터만 추출 및 날짜 정렬
    df = df_master[df_master['채수위치'] == station].copy()
    if len(df) == 0:
        print(f"⚠️ '{station}' 데이터가 없어 건너뜁니다.")
        continue
        
    df['조사일'] = pd.to_datetime(df['조사일'])
    df = df.sort_values('조사일')
    
    # 💡 [핵심] 지점별 변수(TN, TP) 분기 처리
    if station == '문의':
        columns_X = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            'TN(㎎/L)', 'TP(㎎/L)', '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 
            '유해남조류 세포수 (cells/㎖)' # 오늘 조류 세포수 (과거 힌트)
        ]
    else: # 추동, 회남
        columns_X = [
            '수온(℃)', 'pH', 'DO(㎎/L)', '투명도', '탁도', 'Chl-a (㎎/㎥)', 
            '평균기온(°C)', '일강수량(mm)', '합계 일조시간(hr)', '합계 일사량(MJ/m2)', 
            '수위(EL.m)', '저수량(백만㎥)', 
            '유입량(㎥/s)', '총방류량(㎥/s)', 
            '유해남조류 세포수 (cells/㎖)' # 오늘 조류 세포수 (과거 힌트)
        ]
        
    # =====================
    # 3. 7일 선행 시간 매칭 (0.92의 영광을 만든 완벽한 로직 복구!)
    # =====================
    df_clean = df.dropna(subset=columns_X + ['조사일']).copy()

    df_target = df_clean[['조사일', '유해남조류 세포수 (cells/㎖)']].copy()
    df_target = df_target.rename(columns={'조사일': '타겟_조사일', '유해남조류 세포수 (cells/㎖)': 'Target_7일뒤_세포수'})

    df_clean['예측목표일'] = df_clean['조사일'] + pd.Timedelta(days=7)

    df_clean = df_clean.sort_values('예측목표일')
    df_target = df_target.sort_values('타겟_조사일')

    df_merged = pd.merge_asof(
        df_clean, 
        df_target,
        left_on='예측목표일',
        right_on='타겟_조사일',
        direction='nearest',
        tolerance=pd.Timedelta(days=2) # 오차 2일 허용
    )

    df_final = df_merged.dropna(subset=['Target_7일뒤_세포수']).copy()
    print(f"✅ 7일 선행 데이터 매칭 완료! ({station} 지점 학습 데이터: {len(df_final)}행)")
    
    if len(df_final) < 20:
        print(f"⚠️ 학습 데이터 부족으로 [{station}] 지점은 건너뜁니다.")
        continue

    # =====================
    # 4. 학습 준비 및 스케일링
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
    # 5. 인공신경망 학습
    # =====================
    class DynamicANN(nn.Module):
        def __init__(self, input_dim, hidden_layers, input_size, activation_fn):
            super().__init__()
            layers = []
            in_dim = input_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(in_dim, input_size))
                layers.append(nn.BatchNorm1d(input_size))
                layers.append(activation_fn)
                in_dim = input_size
            layers.append(nn.Linear(in_dim, 1))
            self.model = nn.Sequential(*layers)
        def forward(self, x): return self.model(x)

    hidden_layer_options = [3, 4]
    input_size_options = [64, 128]
    activation_options = ['relu', 'tanh']
    epoch_options = [400, 600]
    train_test_split_options = [0.8, 0.9]
    
    results = []
    best_r2 = -float('inf')
    best_model_data = None
    best_plot_data = None # 시각화를 위한 정답/예측값 보관용
    
    for hl, isize, act_name, ep, tr in itertools.product(
        hidden_layer_options, input_size_options, activation_options, epoch_options, train_test_split_options):
        
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        for c in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster']==c]
            if len(cluster_data) < 3: continue 
            n_train = int(len(cluster_data) * tr)
            cluster_train = cluster_data.sample(n=n_train, random_state=42)
            cluster_test = cluster_data.drop(cluster_train.index)
            
            X_train_list.append(cluster_train.drop(columns=['y','cluster']).values)
            X_test_list.append(cluster_test.drop(columns=['y','cluster']).values)
            y_train_list.append(cluster_train['y'].values.reshape(-1,1))
            y_test_list.append(cluster_test['y'].values.reshape(-1,1))
            
        X_train, X_test = np.vstack(X_train_list), np.vstack(X_test_list)
        y_train, y_test = np.vstack(y_train_list), np.vstack(y_test_list)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        
        act_fn = nn.ReLU() if act_name == 'relu' else nn.Tanh()
        model = DynamicANN(X_train.shape[1], hl, isize, act_fn)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        for epoch in range(ep):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()
            
            model.eval()
            val_loss = criterion(model(X_test_t).detach(), y_test_t)
            scheduler.step(val_loss)
            
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).numpy()
            y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)
            y_orig = scaler_y.inverse_transform(y_test)
            
            # 마이너스 값 방지
            y_pred_orig = np.maximum(0, y_pred_orig)
            y_orig = np.maximum(0, y_orig)
            
            r2 = r2_score(y_orig, y_pred_orig)
            mae = mean_absolute_error(y_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_orig, y_pred_orig))
            
        results.append({
            'hidden_layers': hl, 'input_size': isize, 'activation': act_name,
            'epochs': ep, 'train_ratio': tr, 'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2
        })
        
        # 최적 모델 저장 및 그래프 데이터 갱신
        if r2 > best_r2:
            best_r2 = r2
            best_model_data = {
                'state_dict': model.state_dict(),
                'config': {'input_dim': X_train.shape[1], 'hl': hl, 'isize': isize, 'act': act_name}
            }
            best_plot_data = (y_orig.flatten(), y_pred_orig.flatten())
            
        print(f"  [+] R2: {r2:.4f} (MAE: {mae:.0f}) | HL:{hl}, IS:{isize}, EP:{ep}")

    # =====================
    # 6. 폴더별 저장 및 시각화(산점도) 그리기
    # =====================
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(station_dir, "Result_7Days_Ahead.csv"), index=False, encoding='utf-8-sig')
    
    torch.save(best_model_data, os.path.join(station_dir, "best_7days_model.pth"))
    joblib.dump(scaler_X, os.path.join(station_dir, "scaler_X_7days.pkl"))
    joblib.dump(scaler_y, os.path.join(station_dir, "scaler_y_7days.pkl"))
    joblib.dump(kmeans, os.path.join(station_dir, "kmeans_7days.pkl"))
    
    # 📊 실측치 vs 예측치 산점도 그래프 생성
    y_true, y_pred = best_plot_data
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, color='dodgerblue', edgecolors='white', s=50)
    
    max_val = max(np.max(y_true), np.max(y_pred)) if len(y_true) > 0 else 1000
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='완벽한 예측 (y=x)')
    
    plt.xlabel('실제 조류 세포수 (cells/㎖)', fontsize=12)
    plt.ylabel('AI 예측 조류 세포수 (cells/㎖)', fontsize=12)
    plt.title(f'[{station}] 7일 선행 조기경보 AI 성능 (최적 R²: {best_r2:.4f})', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(station_dir, "실측_vs_예측_산점도.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"💾 [{station}] 모델 및 시각화 그래프({plot_path}) 저장 완료!\n")

print("✨ 모든 지역(문의, 추동, 회남)의 맞춤형 모델 생성이 완료되었습니다!")