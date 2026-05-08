import pandas as pd
import glob
import os

# 1. TN/TP 파일들이 모여있는 폴더 경로 지정 (r 붙이는 거 잊지 말고!)
# (예시 경로니까 근서 PC의 실제 폴더 경로로 꼭 수정해 줘!)
folder_path = r'C:\Users\pc\OneDrive\바탕 화면\유근서\.2026 기후부 ax 아이디어 경진대회\ax 공모전 파이썬\2026-AX-Contest-Water\data\tntp'

# 2. 폴더 내의 모든 csv 파일 목록 가져오기
# (파일 이름이 'TNTP 15.01-15.12.csv' 식이므로 *.csv 로 싹 긁어옴)
file_list = glob.glob(os.path.join(folder_path, '*.csv'))
file_list.sort()

print(f"총 {len(file_list)}개의 파일을 찾았습니다. 병합 및 전처리를 시작합니다...\n")

df_list = []
for file in file_list:
    try:
        # 물환경정보시스템 데이터는 보통 utf-8 이지만 혹시 몰라 예외처리!
        df_temp = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        df_temp = pd.read_csv(file, encoding='cp949')
    
    df_list.append(df_temp)

if len(df_list) > 0:
    # 3. 10년 치 데이터를 일단 하나로 쭉 합치기
    df_total = pd.concat(df_list, ignore_index=True)
    
    # 4. [환경공학 필살기] '상층부' 데이터만 필터링!
    # (결측치 에러 방지를 위해 na=False 옵션 추가)
    df_total = df_total[df_total['회차'].str.contains('상층부', na=False)].copy()
    
    # 5. 지점명 매핑 (대청댐1 -> 문의, 대청댐2 -> 추동, 대청댐4 -> 회남)
    mapping_dict = {
        '대청댐1': '문의',
        '대청댐2': '추동',
        '대청댐4': '회남'
    }
    df_total['채수위치'] = df_total['측정소명'].map(mapping_dict)
    
    # 매핑되지 않은 나머지 지점(대청댐3, 5, 6 등)은 과감히 버림!
    df_total = df_total.dropna(subset=['채수위치']).copy()
    
    # 6. 날짜 형식 변환 및 최종 컬럼 정리
    df_total['조사일'] = pd.to_datetime(df_total['년/월/일'])
    
    # 우리에게 딱 필요한 4개 컬럼만 남기기
    df_final = df_total[['조사일', '채수위치', 'TN(㎎/L)', 'TP(㎎/L)']]
    
    # 날짜와 지점명 순서대로 예쁘게 정렬
    df_final = df_final.sort_values(by=['조사일', '채수위치']).reset_index(drop=True)
    
    # 7. 결과 확인
    print(f"✅ 상층부 추출 및 지점명 매핑 완료! (총 데이터 수: {len(df_final)}건)")
    print(df_final.head())
    
    # 8. 다음 번에 바로 쓸 수 있게 깔끔한 통합본 CSV로 저장!
    save_path = os.path.join(folder_path, '대청호_TN_TP_통합본_2015_2026.csv')
    df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 통합본 저장 완료: {save_path}")
else:
    print("❌ CSV 파일을 찾지 못했습니다. 폴더 경로를 다시 한번 확인해 주세요!")