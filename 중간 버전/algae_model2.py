import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =====================
# 1. 데이터 로드 및 전처리
# =====================
df = pd.read_csv('대청호_ML_최종_학습데이터_일단위.csv', index_col=0)
df = pd.get_dummies(df, columns=['채수위치']) # 지역명을 숫자로 변환

target_col = '유해남조류 세포수 (cells/㎖)'
y = df[target_col].values
X_df = df.drop(columns=[target_col])
X = X_df.values
feature_names = X_df.columns.tolist() + ['수질_클러스터(패턴)'] # 변수 이름 저장

# =====================
# 2. 시계열 80:20 분할 및 스케일링
# =====================
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# 3. 🌟 하이브리드: K-Means 클러스터링 추가
# =====================
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)

X_train_final = np.hstack((X_train, train_clusters)) # 원본 값에 클러스터 번호만 추가
X_test_final = np.hstack((X_test, test_clusters))

# =====================
# 4. 랜덤 포레스트 모델 학습
# =====================
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_final, y_train)

# =====================
# 5. 성능 평가
# =====================
y_pred = rf_model.predict(X_test_final)
r2 = r2_score(y_test, y_pred)

def get_alert_level(cells):
    if cells < 1000: return 0
    elif cells < 10000: return 1
    elif cells < 1000000: return 2
    else: return 3

y_test_levels = [get_alert_level(val) for val in y_test]
y_pred_levels = [get_alert_level(val) for val in y_pred]
accuracy = accuracy_score(y_test_levels, y_pred_levels)

print("\n🎯 [RF + 클러스터링 하이브리드 모델 성능]")
print(f"✔️ R² Score: {r2 * 100:.2f}%")
print(f"✔️ Alert Accuracy (경보 적중률): {accuracy * 100:.2f}%")

# =====================
# 6. 보고서용 특성 중요도(Feature Importance) 추출 및 시각화
# =====================
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10) # 상위 10개만!

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('녹조 대발생 핵심 원인 분석 (상위 10개 특성 중요도)', fontsize=16, fontweight='bold')
plt.xlabel('중요도 (기여도)', fontsize=12)
plt.ylabel('환경 변수', fontsize=12)
plt.tight_layout()
plt.show()