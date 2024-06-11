# 데이터 정보

1. open_source_asset_pricing.csv

   - Chen and Zimmermann (2021) 논문에서 제공하는 open source 예측인자 데이터 베이스
   - 데이터 기간은 1980년 1월에서 2023년 12월까지
   - 자세한 정보는 open_source_asset_pricing_info.csv 참조

2. welch_goyal.csv

   - Welch and Goyal (2008) 논문에서 사용한 데이터를 포함하는 데이터 파일
   - 데이터 기간은 1926년 12월에서 2022년 12월까지
   - 자세한 정보는 macro_info.csv 참조

3. FRED_MD.csv

   - FRED에서 제공하는 데이터로 구성한 monthly (MD) 거시경제 데이터
   - 데이터 기간은 1959년 1월에서 2023년 12월까지
   - 자세한 정보는 macro_info.csv 참조

## task

데이터 분석

1. 데이터 기초 통계 산출:

   - 회사별로 각 predictor 들의 기초 통계 산출
   - 결측치 개수, 평균, 분산, 첨도, 왜도, 최대 최소값 등
   - 그 외에도 다양한 유의한 통계치들을 산출한 후 회사별로 정리

2. 데이터 시각화

   - 데이터를 직관적으로 파악할 수 있는 시각화 자료 산출
   - histogram 등을 통해 데이터의 분포를 확인
   - scatterplot으로 target (asset return)과의 선형적인 관계를 파악
   - 이외에 추후 머신러닝 분석에 도움이 될만한 도식 도출

위의 일부를 잘 정리하여 최종 발표때 한 두페이지 분량으로 발표할 수 있도록 함
