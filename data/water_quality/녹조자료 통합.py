import os
import glob
import pandas as pd

# ==========================================
# 🌟 파이썬 파일이 있는 곳을 강제로 기준 폴더로 설정!
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"🚀 작업 폴더: {current_dir}")
print("📊 연도별 녹조자료(엑셀) 10년 치 자동 병합을 시작합니다...\n")

# ==========================================
# 1. '녹조자료'라는 이름이 들어간 모든 엑셀 파일 찾기
# ==========================================
file_list = glob.glob('*녹조자료*.xlsx')
file_list.sort() # 연도별로 예쁘게 정렬

if len(file_list) == 0:
    print("❌ 에러: 폴더 안에 '녹조자료'가 포함된 엑셀(.xlsx) 파일이 없습니다!")
    exit()

print(f"총 {len(file_list)}개의 녹조자료 파일을 찾았습니다. 병합 중...")

df_list = []
for file in file_list:
    try:
        # 🚨 핵심 방어 로직: 물환경정보시스템 엑셀은 위 2줄이 병합된 쓰레기 행이므로 건너뜀 (skiprows=2)
        df_temp = pd.read_excel(file, skiprows=2)
        df_list.append(df_temp)
        print(f"  ✔️ {file} 읽기 완료!")
    except Exception as e:
        print(f"  ⚠️ {file} 읽기 실패: {e}")

# ==========================================
# 2. 데이터 위아래로 한 번에 합치기
# ==========================================
if len(df_list) > 0:
    df_total = pd.concat(df_list, ignore_index=True)

    # 3. 날짜 형식 깔끔하게 통일 (예: 2017.08.07 -> 2017-08-07)
    df_total['조사일'] = pd.to_datetime(df_total['조사일'], format='%Y.%m.%d', errors='coerce').dt.strftime('%Y-%m-%d')

    # 불필요한 빈 행(조사일이 없는 행) 제거
    df_total = df_total.dropna(subset=['조사일'])

    # 4. 최종 마스터 CSV 파일로 저장
    output_name = '대청댐_녹조자료_2015_2026.csv'
    df_total.to_csv(output_name, index=False, encoding='utf-8-sig')
    
    print(f"\n🎯 10년 치 녹조자료 병합 완료! 총 {len(df_total)}행의 '{output_name}' 파일이 생성되었습니다!")