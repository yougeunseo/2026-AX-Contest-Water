import pandas as pd
import os

# 1. 원본 마스터 데이터 불러오기
file_path = '대청호_ML_최종_학습데이터_일단위.csv'
print("⏳ 마스터 데이터를 불러오는 중...")

try:
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949')
    print(f"✅ 데이터 로드 완료! (총 {len(df)}행)")
except FileNotFoundError:
    print("❌ 에러: 원본 CSV 파일이 같은 폴더에 있는지 확인해 줘!")
    exit()

# 2. 제거해야 할 영양염류 파생 변수 리스트
drop_columns = ['TN(㎎/L)', 'TP(㎎/L)', 'NP_Ratio']

# ==========================================
# 3. 지역별로 쪼개고 저장하기
# ==========================================

# 🟢 [문의] 수역: 모든 변수(TN, TP 포함) 살림
df_muni = df[df['채수위치'] == '문의'].copy()
muni_filename = '대청호_문의_학습데이터.csv'
df_muni.to_csv(muni_filename, index=False, encoding='utf-8-sig')
print(f"✅ [문의] 파일 생성 완료! ({len(df_muni)}행, 컬럼 {len(df_muni.columns)}개) -> TN, TP 유지됨")

# 🔵 [추동] 수역: TN, TP, NP_Ratio 열(Column) 과감하게 삭제!
df_chudong = df[df['채수위치'] == '추동'].copy()
df_chudong = df_chudong.drop(columns=drop_columns, errors='ignore')
chudong_filename = '대청호_추동_학습데이터.csv'
df_chudong.to_csv(chudong_filename, index=False, encoding='utf-8-sig')
print(f"✅ [추동] 파일 생성 완료! ({len(df_chudong)}행, 컬럼 {len(df_chudong.columns)}개) -> TN, TP 삭제됨")

# 🔴 [회남] 수역: TN, TP, NP_Ratio 열(Column) 과감하게 삭제!
df_hoenam = df[df['채수위치'] == '회남'].copy()
df_hoenam = df_hoenam.drop(columns=drop_columns, errors='ignore')
hoenam_filename = '대청호_회남_학습데이터.csv'
df_hoenam.to_csv(hoenam_filename, index=False, encoding='utf-8-sig')
print(f"✅ [회남] 파일 생성 완료! ({len(df_hoenam)}행, 컬럼 {len(df_hoenam.columns)}개) -> TN, TP 삭제됨")

print("\n🎉 작업 끝! 폴더를 확인해 보면 3개의 새로운 CSV 파일이 예쁘게 만들어져 있을 거야.")