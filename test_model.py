import dill 
import pandas as pd 
import numpy as np

MODEL_PATH = 'models_final/hybrid_system.pkl'

print(f"'{MODEL_PATH}' 하이브리드 시스템을 로드합니다...")
try:
    with open(MODEL_PATH, 'rb') as f:
        hybrid_system = dill.load(f)
    print("시스템 로드 성공.")
except Exception as e:
    print(f"시스템 로드 실패: {e}")
    exit()

# --- 2. 입력 데이터 정의 ---
# symptom_input_dict = {
#     'cough': 1,
#     'sore throat': 1,
#     'nasal congestion': 1
# }
# (Common Cold 또는 Flu 증상 예시)

symptom_input_dict = {
    'anxiety and nervousness': 1,
    'depression': 1,
    'insomnia': 1
}


# --- 3. 예측 및 추천 수행 ---
try:
    print(f"\n입력 증상: {list(symptom_input_dict.keys())}")
    
    result_memory = hybrid_system.predict_and_recommend(
        symptom_input_dict, 
        n_recommendations=3, 
        method='memory'
    )
    
    result_content = hybrid_system.predict_and_recommend(
        symptom_input_dict, 
        n_recommendations=3, 
        method='content'
    )

    print("\n" + "="*50)
    print(" [1] 메모리 기반 추천 (평점/리뷰 수 기준)")
    print("="*50)
    print(f"  > 예측된 질병: {result_memory['predicted_disease']}")
    print("  > 추천 약물:")
    print(result_memory['recommendations'])

    print("\n" + "="*50)
    print(" [2] 콘텐츠 기반 추천 (리뷰 텍스트 키워드 기준)")
    print("="*50)
    print(f"  > 예측된 질병: {result_content['predicted_disease']}")
    print("  > 추천 약물:")
    print(result_content['recommendations'])

except Exception as e:
    print(f"\n예측/추천 실패: {e}")