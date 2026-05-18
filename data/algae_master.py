import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# 폰트 설정 (Windows: Malgun Gothic)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# [STEP 1] 데이터 로드 및 "다이어트(Feature Selection)"
# ==========================================
# 🚨 아까 빠졌던 load_data 함수 부활!
def load_data(file_name):
    try:
        return pd.read_csv(file_name, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(file_name, encoding='cp949')

df = load_data('대청호_ML_최종_학습데이터_일단위.csv')
if '조사일' in df.columns:
    df = df.set_index('조사일')

# 채수위치 원핫 인코딩 (문의, 추동, 회남 구분)
df = pd.get_dummies(df, columns=['채수위치'])

# 타겟 데이터 분리
target_col = '유해남조류 세포수 (cells/㎖)'
y = df[target_col].values

# 🚨 웹사이트(대시보드)와 똑같이 13개의 핵심 변수만 추려내기!
# (주의: 아래 컬럼명들이 실제 CSV 파일의 이름과 띄어쓰기까지 똑같아야 해)
core_columns = [
    '수온(℃)', 'DO(㎎/L)', '투명도', '일강수량(mm)', 'Chl-a (㎎/㎥)', 
    'TP(㎎/L)', 'TN(㎎/L)', 'NP_Ratio', 'Retention_Index', '총방류량(㎥/s)',
    '채수위치_문의', '채수위치_추동', '채수위치_회남'
]

# 불필요한 날씨/기압 데이터는 버리고 핵심 13개 변수만 남김
X_df = df[core_columns] 
X = X_df.values
feature_names = X_df.columns.tolist() + ['수질_클러스터(패턴)']

# ==========================================
# [STEP 2] 시계열 기반 데이터 분할 (80:20)
# ==========================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 스케일링 (정규화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# [STEP 3] 하이브리드 기법: K-Means 클러스터링
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled).reshape(-1, 1)
test_clusters = kmeans.predict(X_test_scaled).reshape(-1, 1)

# 군집 정보를 새로운 변수로 추가 (13개 + 1개 = 14개 변수)
X_train_final = np.hstack((X_train_scaled, train_clusters))
X_test_final = np.hstack((X_test_scaled, test_clusters))

# ==========================================
# [STEP 4] 랜덤 포레스트 모델 학습
# ==========================================
print("🚀 13개 핵심 변수로 AI 다이어트 모델 학습 중...")
rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_final, y_train)

# ==========================================
# [STEP 5] 성적표 산출 (R2 & 적중률)
# ==========================================
y_pred = rf_model.predict(X_test_final)
y_pred = np.maximum(0, y_pred) # 음수 방지

# 조류경보제 등급 분류
def get_alert_level(cells):
    if cells < 1000: return 0      # 평상시
    elif cells < 10000: return 1   # 관심
    elif cells < 1000000: return 2 # 경계
    else: return 3                 # 대발생

y_test_levels = [get_alert_level(val) for val in y_test]
y_pred_levels = [get_alert_level(val) for val in y_pred]

r2 = r2_score(y_test, y_pred)
acc = accuracy_score(y_test_levels, y_pred_levels)

print(f"\n✅ 분석 결과 요약 (다이어트 모델)")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  - 수치 설명력 (R²): {r2 * 100:.2f}%")
print(f"  - 경보 적중률 (Accuracy): {acc * 100:.2f}%")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ==========================================
# [STEP 6] 모델 파일(.pkl) 저장
# ==========================================
# 이제 이 파일들은 24개가 아닌 13개 변수에 완벽하게 맞춰진 뇌가 됩니다.
joblib.dump(rf_model, 'final_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans.pkl')
print("\n💾 13개 변수 전용 모델(.pkl)이 성공적으로 덮어쓰기 되었습니다!")