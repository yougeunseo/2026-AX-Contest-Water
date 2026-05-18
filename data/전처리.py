import pandas as pd
import numpy as np

# 🌟 인코딩 에러 방어용 만능 로드 함수
def load_csv_safe(file_name):
    try:
        return pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_name, encoding='cp949')
        except UnicodeDecodeError:
            return pd.read_csv(file_name, encoding='euc-kr')

print("⏳ 원본 데이터 4종을 불러오는 중...")
df_algae = load_csv_safe('대청댐_녹조자료_2015_2026.csv')
df_tntp = load_csv_safe('대청댐_TNTP_2015_2026.csv')
df_water = load_csv_safe('대청댐_수위자료_2015_2026.csv')
df_weather = load_csv_safe('대청댐_기상자료_2015_2026.csv')

# ==========================================
# 1. 날짜 데이터 형식 통일 (조사일 / 일시)
# ==========================================
df_algae['조사일'] = pd.to_datetime(df_algae['조사일'])
df_tntp['조사일'] = pd.to_datetime(df_tntp['조사일'])
df_water['일시'] = pd.to_datetime(df_water['일시'])
df_weather['일시'] = pd.to_datetime(df_weather['일시'])

# 병합을 위해 이름 통일 ('일시' -> '조사일')
df_water = df_water.rename(columns={'일시': '조사일'})
df_weather = df_weather.rename(columns={'일시': '조사일'})

# ==========================================
# 2. 🚨 핵심! 조류 측정 안 한 날짜 & 불필요한 열 과감히 날리기
# ==========================================
target_col = '유해남조류 세포수 (cells/㎖)'
# 정답(세포수)이 없는 행은 얄짤없이 삭제
df_algae_clean = df_algae.dropna(subset=[target_col]).copy()

# 쓸데없는 꼬리표(우점종, 독소, 냄새 등) 열 찾아서 삭제
drop_keywords = ['우점종', '냄새물질', '조류독소', '분류', '지점명']
cols_to_drop = [col for col in df_algae_clean.columns if any(kw in col for kw in drop_keywords)]
df_algae_clean = df_algae_clean.drop(columns=cols_to_drop, errors='ignore')

# 기상자료에서도 중복되는 '지점' 관련 글자 삭제
if '지점' in df_weather.columns:
    df_weather = df_weather.drop(columns=['지점', '지점명'], errors='ignore')

# ==========================================
# 3. 지역별 맞춤형 데이터 조립 함수 (TN/TP 분리)
# ==========================================
def process_region(region_name, keep_tntp=False):
    # 해당 지역의 세포수 데이터만 필터링
    region_df = df_algae_clean[df_algae_clean['채수위치'] == region_name].copy()
    
    # 조류 측정일(조사일)을 기준으로 수위자료, 기상자료 갖다 붙이기 (Left Join)
    region_df = pd.merge(region_df, df_water, on='조사일', how='left')
    region_df = pd.merge(region_df, df_weather, on='조사일', how='left')
    
    # 🟢 문의 지역일 경우: TN, TP 붙이고 N/P 비율 계산
    if keep_tntp:
        tntp_reg = df_tntp[df_tntp['채수위치'] == region_name].copy()
        region_df = pd.merge(region_df, tntp_reg, on=['조사일', '채수위치'], how='left')
        region_df['NP_Ratio'] = region_df['TN(㎎/L)'] / region_df['TP(㎎/L)'].replace(0, np.nan)
        
    # 🌟 공통 필수 파생변수: 체류시간 지표 생성
    region_df['Retention_Index'] = region_df['저수량(백만㎥)'] / region_df['총방류량(㎥/s)'].replace(0, np.nan)
    
    # 날짜순 정렬
    region_df = region_df.sort_values('조사일')
    
    return region_df

# ==========================================
# 4. 세 지역 찢어서 변환 & 저장
# ==========================================
print("🚀 지역별 맞춤형 순수 데이터 조립 중...")
muni_df = process_region('문의', keep_tntp=True)
chudong_df = process_region('추동', keep_tntp=False)
hoenam_df = process_region('회남', keep_tntp=False)

# 최종 엑셀 파일로 굽기
muni_df.to_csv('대청호_문의_순수데이터.csv', index=False, encoding='utf-8-sig')
chudong_df.to_csv('대청호_추동_순수데이터.csv', index=False, encoding='utf-8-sig')
hoenam_df.to_csv('대청호_회남_순수데이터.csv', index=False, encoding='utf-8-sig')

print(f"✅ [문의] 완성! (총 {len(muni_df)}행) - TN, TP, NP_Ratio 포함")
print(f"✅ [추동] 완성! (총 {len(chudong_df)}행) - TN, TP 제외됨")
print(f"✅ [회남] 완성! (총 {len(hoenam_df)}행) - TN, TP 제외됨")
print("🎉 작업 끝! 인공적인 보간 없는 순도 100% 데이터셋이 준비되었습니다.")