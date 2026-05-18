import pandas as pd
import glob
import os

# 1. 파일들이 있는 폴더 경로 (r 붙인 상태 유지, 정확한 폴더까지 지정!)
folder_path = r'C:\Users\pc\OneDrive\바탕 화면\유근서\.2026 기후부 ax 아이디어 경진대회\ax 공모전 파이썬\2026-AX-Contest-Water\data\water_level'

# 2. '수위자료'가 포함된 모든 'xlsx' 파일 목록 가져오기
file_list = glob.glob(os.path.join(folder_path, '*수위자료*.xlsx'))
file_list.sort() # 연도순으로 정렬

print(f"총 {len(file_list)}개의 파일을 찾았습니다. 병합을 시작합니다...\n")

df_list = []
for file in file_list:
    # 🚨 핵심 포인트: 엑셀 파일 읽을 때 위에 2줄(대청댐 어쩌고) 건너뛰기!
    df_temp = pd.read_excel(file, skiprows=2)
    df_list.append(df_temp)

# 파일이 제대로 찾아졌는지 확인 후 병합
if len(df_list) > 0:
    # 3. 데이터프레임 리스트를 위아래로 한 번에 합치기
    df_dam_total = pd.concat(df_list, ignore_index=True)

    # 4. 날짜 형식으로 변환 후 '과거 -> 최신' 순으로 재정렬
    df_dam_total['일시'] = pd.to_datetime(df_dam_total['일시'])
    df_dam_total = df_dam_total.sort_values(by='일시', ascending=True).reset_index(drop=True)

    # 5. 결과 확인
    print(f"✅ 10년 치 수문 자료 병합 및 정렬 완료! (총 데이터 수: {len(df_dam_total)}건)")
    print(df_dam_total.head(3))
    
    # 6. 다음 번에 쓸 수 있게 CSV 통합본으로 저장해두기
    save_path = os.path.join(folder_path, '대청댐_수문자료_통합본_2015_2026.csv')
    df_dam_total.to_csv(save_path, index=False, encoding='utf-8-sig') 
    print(f"\n💾 통합본 저장 완료: {save_path}")
else:
    print("❌ 엑셀 파일을 찾지 못했습니다. 폴더 경로를 다시 한번 확인해 주세요!")