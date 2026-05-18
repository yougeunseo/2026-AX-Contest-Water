import os
import pandas as pd

# ==========================================
# [경로 에러 해결] 파이썬 파일이 있는 곳을 강제로 기준 폴더로 설정
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print(f"🚀 작업 폴더 강제 고정 완료: {current_dir}")
print("📊 3개 지역 순수 데이터 통합 작업을 시작합니다...\n")

# ==========================================
# 1. 3개 지점 순수데이터 불러오기
# ==========================================
try:
    df_muni = pd.read_csv('대청호_문의_순수데이터.csv', encoding='utf-8-sig')
    df_chudong = pd.read_csv('대청호_추동_순수데이터.csv', encoding='utf-8-sig')
    df_hoenam = pd.read_csv('대청호_회남_순수데이터.csv', encoding='utf-8-sig')
    print("✅ 문의, 추동, 회남 순수 데이터 로드 성공!")
except Exception as e:
    print(f"❌ 파일 읽기 실패: {e}")
    print("💡 이 파이썬 파일과 3개의 csv 파일이 같은 폴더에 있는지 확인해주세요!")
    exit()

# ==========================================
# 2. 데이터 위아래로 합치기 (Concat)
# ==========================================
# 3개의 엑셀을 밑으로 길게 이어 붙임 (문의 19개 열, 추동/회남 16개 열이라 빈칸은 알아서 NaN 처리됨)
df_master = pd.concat([df_muni, df_chudong, df_hoenam], ignore_index=True)

# ==========================================
# 3. 날짜 및 지역 순으로 정렬하기
# ==========================================
df_master['조사일'] = pd.to_datetime(df_master['조사일'])
df_master = df_master.sort_values(by=['조사일', '채수위치']).reset_index(drop=True)

# ==========================================
# 4. 최종 통합 마스터 파일로 저장
# ==========================================
output_filename = '대청호_통합_순수데이터.csv'
df_master.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"🎯 통합 완료! 총 {len(df_master)}행의 마스터 데이터 '{output_filename}'가 생성되었습니다!\n")
