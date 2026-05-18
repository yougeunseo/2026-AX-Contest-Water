import pandas as pd
import numpy as np

# 🌟 인코딩 에러 방어 함수
def load_csv_safe(file_name):
    try:
        return pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_name, encoding='cp949')
        except UnicodeDecodeError:
            return pd.read_csv(file_name, encoding='euc-kr')

# 1. 데이터 로드
df_algae = load_csv_safe('대청댐_녹조자료_2015_2026.csv')
df_tntp = load_csv_safe('대청댐_TNTP_2015_2026.csv')
df_water = load_csv_safe('대청댐_수위자료_2015_2026.csv')
df_weather = load_csv_safe('대청댐_기상자료_2015_2026.csv')

# 날짜 형식 변환
for df, col in [(df_algae, '조사일'), (df_tntp, '조사일'), (df_water, '일시'), (df_weather, '일시')]:
    df[col] = pd.to_datetime(df[col])

# --- STEP 1: 지역별 데이터 병합 (녹조 + TNTP) ---
drop_cols = [col for col in df_algae.columns if '우점종' in col or '냄새' in col or '독소' in col or '분류' in col or '지점명' in col]
df_algae = df_algae.drop(columns=drop_cols)
df_merged = pd.merge(df_algae, df_tntp, on=['조사일', '채수위치'], how='outer')

# --- STEP 2: 공통 데이터 병합 (수위 + 기상) ---
df_common = pd.merge(df_water, df_weather, on='일시', how='outer')

# --- STEP 3: 전체 대통합 및 파생변수 생성 ---
df_common = df_common.rename(columns={'일시': '조사일'})
df_final = pd.merge(df_merged, df_common, on='조사일', how='left')

# 파생변수 생성
df_final['NP_Ratio'] = df_final['TN(㎎/L)'] / df_final['TP(㎎/L)'].replace(0, np.nan)
df_final['Retention_Index'] = df_final['저수량(백만㎥)'] / df_final['총방류량(㎥/s)'].replace(0, np.nan)

# --- STEP 4: 시계열 확장 및 보간 (Interpolation) ---
regions = df_final['채수위치'].dropna().unique()
interpolated_list = []

for reg in regions:
    reg_df = df_final[df_final['채수위치'] == reg].copy()
    
    # 🚨 여기가 에러 해결의 핵심! (중복 날짜 처리)
    # 같은 날짜에 측정값이 여러 개면 평균을 내서 하나로 합침
    reg_df = reg_df.groupby('조사일').mean(numeric_only=True)
    
    # 이제 중복이 없으니 안심하고 매일(Daily) 단위로 달력 확장!
    reg_df = reg_df.resample('D').asfreq() 
    reg_df = reg_df.interpolate(method='time') # 빈칸을 선형 보간으로 채움
    reg_df['채수위치'] = reg
    
    # 7일 시프트 (타겟인 세포수만 가만히 두고 나머지를 7일 뒤로 미룸)
    target = reg_df['유해남조류 세포수 (cells/㎖)'].copy()
    features = reg_df.drop(columns=['유해남조류 세포수 (cells/㎖)']).shift(7)
    
    final_reg_df = pd.concat([features, target], axis=1).dropna()
    interpolated_list.append(final_reg_df)

# 최종 합치기
df_ultimate = pd.concat(interpolated_list).sort_index()

# 저장
df_ultimate.to_csv('대청호_ML_최종_학습데이터_일단위.csv', encoding='utf-8-sig')
print("✅ 전처리 완료! 중복 데이터 평균 처리 및 일일 보간 성공!")