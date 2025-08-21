import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 사전 준비: 외부 데이터 로드 및 전처리
def load_and_prepare_lookup_tables():
    print("[INFO] 파생변수 생성을 위한 외부 데이터를 로드하고 전처리합니다...")
    
    sido_map = {
        '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구', '인천광역시': '인천',
        '광주광역시': '광주', '대전광역시': '대전', '울산광역시': '울산', '세종특별자치시': '세종',
        '경기도': '경기', '강원특별자치도': '강원', '충청북도': '충북', '충청남도': '충남',
        '전북특별자치도': '전북', '전라남도': '전남', '경상북도': '경북', '경상남도': '경남',
        '제주특별자치도': '제주'
    }

    try:
        full_df = pd.read_csv('./data/전세보증_데이터_최종결합본.csv', encoding='cp949') 
        full_df['선순위'] = pd.to_numeric(full_df['선순위'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        avg_lien_by_sido = full_df.groupby('시도')['선순위'].mean().to_dict()
        
        lookup_df = full_df[['시도', '보증시작월', '순이동률(%)', '보증완료월_실업률']].copy()
        lookup_df.rename(columns={'보증시작월': '연도월'}, inplace=True)
        lookup_df = lookup_df.drop_duplicates(subset=['시도', '연도월']).set_index(['시도', '연도월'])
    except Exception as e:
        print(f"Warning: 최종결합본 데이터 로드 실패 - {e}")
        avg_lien_by_sido = {}
        lookup_df = pd.DataFrame()

    try:
        price_index_df = pd.read_csv('./data/주택매매지수.csv', encoding='cp949') 
        price_index_df['시도'] = price_index_df['행정구역별'].map(sido_map).fillna(price_index_df['행정구역별'])
        price_index_df = price_index_df.drop(columns=['행정구역별']).set_index('시도')
        price_index_df.columns = [int(col.replace('.', '')) for col in price_index_df.columns]
    except Exception as e:
        print(f"Warning: 주택매매지수 데이터 로드 실패 - {e}")
        price_index_df = pd.DataFrame()

    try:
        interest_df = pd.read_csv('./data/한국은행_금리.csv', encoding='cp949') 
        interest_df['연도월'] = interest_df['연도'] * 100 + interest_df['월']
        interest_df = interest_df.set_index('연도월')['금리']
        all_months_idx = pd.date_range(start=f'{interest_df.index.min()//100}-01-01', end=f'{interest_df.index.max()//100}-12-31', freq='MS').strftime('%Y%m').astype(int)
        interest_df = interest_df.reindex(all_months_idx).ffill()
    except Exception as e:
        print(f"Warning: 금리 데이터 로드 실패 - {e}")
        interest_df = pd.Series()
        
    print("[INFO] 외부 데이터 준비 완료.")
    return {
        "avg_lien": avg_lien_by_sido, "price_index": price_index_df,
        "interest_rate": interest_df, "econ_lookup": lookup_df
    }

# 파생변수 생성 핵심 함수
def generate_features(input_data: dict, lookup_tables: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])
    house_value = df.at[0, '주택가액'] if df.at[0, '주택가액'] > 0 else 1
    df['초기LTV'] = (df['선순위'] + df['임대보증금액']) / house_value
    df['계산_LTV'] = df['초기LTV']
    df['선순위비율'] = df['선순위'] / house_value
    df['담보여유금액'] = df['주택가액'] - (df['선순위'] + df['임대보증금액'])
    df['잔여가치율'] = (df['주택가액'] - df['선순위']) / house_value
    df['보증금_대비_주택가액_비율'] = df['임대보증금액'] / house_value
    df['선순위_보증금_합계_비율'] = df['초기LTV'] * 100
    start_dt = datetime.strptime(str(df.at[0, '보증시작월']), '%Y%m')
    end_dt = datetime.strptime(str(df.at[0, '보증완료월']), '%Y%m')
    df['보증시작_연도'] = start_dt.year
    df['보증시작_월'] = start_dt.month
    df['보증시작_분기'] = (start_dt.month - 1) // 3 + 1
    df['보증종료_연도'] = end_dt.year
    df['보증종료_월'] = end_dt.month
    df['보증종료_분기'] = (end_dt.month - 1) // 3 + 1
    delta = relativedelta(end_dt, start_dt)
    df['보증기간개월'] = delta.years * 12 + delta.months
    df['경과기간개월'] = 0
    df['잔여기간개월'] = df['보증기간개월']
    month = df.at[0, '보증시작_월']
    df['계절구분_봄'] = 1 if month in [3, 4, 5] else 0
    df['계절구분_여름'] = 1 if month in [6, 7, 8] else 0
    df['계절구분_겨울'] = 1 if month in [12, 1, 2] else 0
    sido = df.at[0, '시도']
    property_type = df.at[0, '주택구분']
    all_sidos = ['경기', '경남', '경북', '광주', '대구', '대전', '부산', '서울', '세종', '울산', '인천', '전남', '전북', '제주', '충남', '충북']
    for s in all_sidos: df[f'시도_{s}'] = 1 if s == sido else 0
    df['지역구분_지방'] = 1 if sido not in ['서울', '경기', '인천'] else 0
    all_property_types = ['다가구주택', '다세대주택', '다중주택', '단독주택', '아파트', '연립주택', '오피스텔', '주상복합']
    for pt in all_property_types: df[f'주택구분_{pt}'] = 1 if pt == property_type else 0
    start_month = df.at[0, '보증시작월']
    end_month = df.at[0, '보증완료월']
    avg_lien = lookup_tables['avg_lien'].get(sido, 1)
    df['지역별_선순위_평균대비_비율'] = df['선순위'] / (avg_lien if avg_lien > 0 else 1)
    price_index_date = int((start_dt - relativedelta(years=1)).strftime('%Y%m'))
    try: df['주택매매지수'] = lookup_tables['price_index'].at[sido, price_index_date]
    except: df['주택매매지수'] = 100
    try: df['보증완료금리'] = lookup_tables['interest_rate'].at[end_month]
    except: df['보증완료금리'] = 3.5
    try:
        df['순이동률(%)'] = lookup_tables['econ_lookup'].at[(sido, start_month), '순이동률(%)']
        df['보증완료월_실업률'] = lookup_tables['econ_lookup'].at[(sido, start_month), '보증완료월_실업률']
    except:
        df['순이동률(%)'] = 0
        df['보증완료월_실업률'] = 3.0
    df = df.rename(columns={'순이동률(%)': '순이동률pct'})
    return df
