import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# =====================
# 1. 데이터 로드 및 지역 인코딩 (One-Hot)
# =====================
file_path = '대청호_ML_최종_학습데이터_일단위.csv'
df = pd.read_csv(file_path, index_col=0)

# '채수위치'(글자)를 AI가 읽을 수 있게 0과 1의 숫자(One-Hot)로 변환!
df = pd.get_dummies(df, columns=['채수위치'])

# 타겟(Y)과 독립변수(X) 분리
target_col = '유해남조류 세포수 (cells/㎖)'
y = df[target_col].values
X = df.drop(columns=[target_col]).values

# =====================
# 2. 시계열 분할 (Time-series Split) 및 정규화
# =====================
# 🚨 랜덤으로 섞지 않고, 시간 순서대로 80% 학습, 20% 테스트 분할
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 스케일링 (수치 범위 통일)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# =====================
# 3. K-Means 클러스터링 (동기의 아이디어 융합!)
# =====================
# 학습 데이터로 수질 패턴(군집)을 3개로 나누고, 이를 새로운 변수로 추가
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)

X_train_final = np.hstack((X_train_scaled, train_clusters))
X_test_final = np.hstack((X_test_scaled, test_clusters))

# 파이토치 텐서로 변환
X_train_t = torch.FloatTensor(X_train_final)
y_train_t = torch.FloatTensor(y_train_scaled).view(-1, 1)
X_test_t = torch.FloatTensor(X_test_final)

# =====================
# 4. 딥러닝 모델 설계 및 학습
# =====================
class AlgaeNet(nn.Module):
    def __init__(self, input_dim):
        super(AlgaeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

model = AlgaeNet(input_dim=X_train_final.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("🚀 딥러닝 모델 학습 시작...")
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# =====================
# 5. 평가 (R2 및 조류경보제 적중률)
# =====================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

# 음수 예측값은 0으로 처리 (세포수가 마이너스일 순 없으므로)
y_pred = np.maximum(0, y_pred)

r2 = r2_score(y_test, y_pred)

def get_alert_level(cells):
    if cells < 1000: return 0    # 평상시
    elif cells < 10000: return 1 # 관심
    elif cells < 1000000: return 2 # 경계
    else: return 3               # 대발생

y_test_levels = [get_alert_level(val) for val in y_test]
y_pred_levels = [get_alert_level(val) for val in y_pred]
accuracy = accuracy_score(y_test_levels, y_pred_levels)

print("\n🎯 [최종 하이브리드 모델 성능 평가]")
print(f"✔️ R² Score (수치 예측력): {r2 * 100:.2f}%")
print(f"✔️ Alert Accuracy (경보 적중률): {accuracy * 100:.2f}%")

# 모델 저장 (나중에 실무 프로그램에서 쓰기 위해)
torch.save(model.state_dict(), 'algae_hybrid_model.pth')
print("✅ 모델 저장 완료 ('algae_hybrid_model.pth')")