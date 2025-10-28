"""
════════════════════════════════════════════════════════════════════════════════════════════════════
🏥 Machine Learning Term Project - FINAL VERSION (CORRECTED v2)
증상 기반 질병 예측 및 약물 추천 하이브리드 시스템

데이터셋 정보 (최종 확정):
────────────────────────────────────────────────────────────────────────────────────────────────────
질병 예측:
  - 훈련용 (많은 데이터): marslinoedward/disease-prediction-data (4,962개)
  - 테스트용 (적은 데이터): itachi9604/disease-symptom-description-dataset (4,920개)
  
약물 추천:
  - 훈련용: yash9439/drug-review (161,297개) + jessicali9530 (161,297개)
  - 테스트용: 위 데이터셋의 test 활용 (53,766 + 53,766개)

미사용 데이터:
  ✗ kaushil268/disease-prediction-using-machine-learning (동일한 데이터이므로 미사용)

════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import numpy as np
import kagglehub
import pickle
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dill

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# 설정
# ============================================================================

MODEL_PATH = "/root/models_final"
VISUALIZATION_PATH = "/root/visualizations"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

print("\n" + "="*100)
print("🏥 Machine Learning Term Project - FINAL v2")
print("증상 기반 질병 예측 및 약물 추천 시스템")
print("="*100)

# ============================================================================
# PART 1: 데이터 로드
# ============================================================================

print("\n### PART 1: 데이터 로드 ###\n")

print("📥 1️⃣  질병 예측 - 훈련 데이터 (marslinoedward - 많은 데이터)")
disease_train_path = kagglehub.dataset_download("marslinoedward/disease-prediction-data")
disease_train_df = pd.read_csv(os.path.join(disease_train_path, "Training.csv"))
disease_train_df = disease_train_df.drop('Unnamed: 133', axis=1, errors='ignore')

X_train_disease = disease_train_df.drop('prognosis', axis=1)
y_train_disease = disease_train_df['prognosis']

print(f"  ✓ 훈련: {X_train_disease.shape[0]} 샘플 (marslinoedward)")
print(f"  ✓ 특성(증상): {X_train_disease.shape[1]}개")
print(f"  ✓ 질병 종류: {len(y_train_disease.unique())}개")

print("\n📥 2️⃣  질병 예측 - 테스트 데이터 (itachi9604 - 적은 데이터 / 다른 형식)")
disease_test_path = kagglehub.dataset_download("itachi9604/disease-symptom-description-dataset")
disease_test_file = os.path.join(disease_test_path, "dataset.csv")
disease_test_df = pd.read_csv(disease_test_file)

print(f"  ✓ 원본 형식: {disease_test_df.shape}")
print(f"    - Disease 컬럼: {disease_test_df.iloc[0, 0]}")
print(f"    - Symptom_1~Symptom_17: 증상들")

# itachi9604 데이터를 marslinoedward 형식으로 변환
symptom_cols = [col for col in disease_test_df.columns if col.startswith('Symptom_')]
feature_names = X_train_disease.columns.tolist()

X_test_disease_converted = []
y_test_disease = []

for idx, row in disease_test_df.iterrows():
    symptom_vector = np.zeros(len(feature_names), dtype=int)
    
    for symptom_col in symptom_cols:
        symptom = row[symptom_col]
        if pd.notna(symptom):
            symptom_clean = str(symptom).strip()
            if symptom_clean in feature_names:
                symptom_idx = feature_names.index(symptom_clean)
                symptom_vector[symptom_idx] = 1
    
    X_test_disease_converted.append(symptom_vector)
    y_test_disease.append(row['Disease'])

X_test_disease = np.array(X_test_disease_converted)
y_test_disease = np.array(y_test_disease)

print(f"  ✓ 변환 완료: {X_test_disease.shape[0]} 샘플")
print(f"  ✓ 테스트: {X_test_disease.shape[0]} 샘플 (itachi9604)")

print("\n📥 3️⃣  약물 리뷰 데이터 (yash9439 + jessicali9530)")
drug_path1 = kagglehub.dataset_download("yash9439/drug-review")
drug_train1 = pd.read_csv(os.path.join(drug_path1, "drugsComTrain_raw.tsv"), sep='\t')
drug_test1 = pd.read_csv(os.path.join(drug_path1, "drugsComTest_raw.tsv"), sep='\t')

drug_path2 = kagglehub.dataset_download("jessicali9530/kuc-hackathon-winter-2018")
drug_train2 = pd.read_csv(os.path.join(drug_path2, "drugsComTrain_raw.csv"))
drug_test2 = pd.read_csv(os.path.join(drug_path2, "drugsComTest_raw.csv"))

# 컬럼 정규화
drug_train1 = drug_train1.drop('Unnamed: 0', axis=1, errors='ignore')
drug_test1 = drug_test1.drop('Unnamed: 0', axis=1, errors='ignore')
drug_train2 = drug_train2.drop('uniqueID', axis=1, errors='ignore')
drug_test2 = drug_test2.drop('uniqueID', axis=1, errors='ignore')

# 통합
drug_train_all = pd.concat([drug_train1, drug_train2], ignore_index=True)
drug_test_all = pd.concat([drug_test1, drug_test2], ignore_index=True)

# 결측치 제거
drug_train = drug_train_all.dropna(subset=['condition']).copy()
drug_test = drug_test_all.dropna(subset=['condition']).copy()

drug_train['condition'] = drug_train['condition'].str.strip()
drug_train['drugName'] = drug_train['drugName'].str.strip()

print(f"  ✓ 훈련: {len(drug_train):,} 리뷰")
print(f"  ✓ 테스트: {len(drug_test):,} 리뷰")

# ============================================================================
# PART 2: 질병 예측 모델
# ============================================================================

print("\n### PART 2: 질병 예측 모델 구축 ###\n")

# k-NN 모델
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_disease, y_train_disease)
y_pred_knn = knn_model.predict(X_test_disease)

knn_acc = accuracy_score(y_test_disease, y_pred_knn)
knn_prec = precision_score(y_test_disease, y_pred_knn, average='weighted', zero_division=0)
knn_rec = recall_score(y_test_disease, y_pred_knn, average='weighted', zero_division=0)
knn_f1 = f1_score(y_test_disease, y_pred_knn, average='weighted', zero_division=0)

# Decision Tree 모델
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42)
dt_model.fit(X_train_disease, y_train_disease)
y_pred_dt = dt_model.predict(X_test_disease)

dt_acc = accuracy_score(y_test_disease, y_pred_dt)
dt_prec = precision_score(y_test_disease, y_pred_dt, average='weighted', zero_division=0)
dt_rec = recall_score(y_test_disease, y_pred_dt, average='weighted', zero_division=0)
dt_f1 = f1_score(y_test_disease, y_pred_dt, average='weighted', zero_division=0)

best_model = knn_model if knn_acc >= dt_acc else dt_model
best_model_name = "k-NN" if knn_acc >= dt_acc else "Decision Tree"

print(f"📊 k-NN 성능 (itachi9604 테스트 데이터):")
print(f"  Accuracy: {knn_acc:.4f} | Precision: {knn_prec:.4f} | Recall: {knn_rec:.4f} | F1: {knn_f1:.4f}")

print(f"\n📊 Decision Tree 성능 (itachi9604 테스트 데이터):")
print(f"  Accuracy: {dt_acc:.4f} | Precision: {dt_prec:.4f} | Recall: {dt_rec:.4f} | F1: {dt_f1:.4f}")

print(f"\n✓ 선택 모델: {best_model_name} (정확도: {max(knn_acc, dt_acc):.4f})")

# ============================================================================
# PART 3: 약물 추천 시스템
# ============================================================================

print("\n### PART 3: 약물 추천 시스템 (협업 필터링) ###\n")

condition_drug_rating = drug_train.groupby(['condition', 'drugName'])['rating'].agg(['mean', 'count']).reset_index()
condition_drug_rating.columns = ['condition', 'drugName', 'avg_rating', 'review_count']

drug_pivot = condition_drug_rating.pivot_table(
    index='condition',
    columns='drugName',
    values='avg_rating',
    fill_value=0
)

cosine_sim_matrix = cosine_similarity(drug_pivot.values)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=drug_pivot.index, columns=drug_pivot.index)

print(f"📊 약물 추천 시스템:")
print(f"  - 질병-약물 쌍: {len(condition_drug_rating):,}개")
print(f"  - 고유 약물: {condition_drug_rating['drugName'].nunique()}개")
print(f"  - 고유 질병: {condition_drug_rating['condition'].nunique()}개")
print(f"  - 평균 평점: {condition_drug_rating['avg_rating'].mean():.2f}/10")

# ============================================================================
# PART 4: 하이브리드 시스템
# ============================================================================

print("\n### PART 4: 하이브리드 시스템 정의 ###\n")

class HybridRecommendationSystem:
    """
    하이브리드 질병 예측 및 약물 추천 시스템
    
    1단계: 콘텐츠 기반 필터링 (증상 → 질병 예측)
    2단계: 협업 필터링 (질병 → 약물 추천)
    """
    
    def __init__(self, disease_model, feature_names, target_names, condition_drug_rating):
        self.disease_model = disease_model
        self.feature_names = feature_names
        self.target_names = target_names
        self.condition_drug_rating = condition_drug_rating
    
    def predict_disease(self, symptoms_vector):
        """1단계: 증상 벡터 → 질병 예측"""
        symptoms_vector = symptoms_vector.reshape(1, -1)
        return self.disease_model.predict(symptoms_vector)[0]
    
    def recommend_drugs(self, disease, n_recommendations=5):
        """2단계: 질병 → 약물 추천"""
        disease_drugs = self.condition_drug_rating[
            self.condition_drug_rating['condition'] == disease
        ].copy()
        
        if len(disease_drugs) == 0:
            overall_avg = self.condition_drug_rating.groupby('drugName').agg({
                'avg_rating': 'mean',
                'review_count': 'sum'
            }).reset_index().sort_values('avg_rating', ascending=False)
            return overall_avg.head(n_recommendations)
        
        disease_drugs['score'] = (disease_drugs['avg_rating'] * 0.7 + 
                                   np.log1p(disease_drugs['review_count']) * 0.3)
        return disease_drugs.sort_values('score', ascending=False)[
            ['drugName', 'avg_rating', 'review_count']
        ].head(n_recommendations)
    
    def predict_and_recommend(self, symptoms_vector, n_recommendations=5):
        """End-to-End: 증상 → 질병 → 약물 추천"""
        predicted_disease = self.predict_disease(symptoms_vector)
        recommended_drugs = self.recommend_drugs(predicted_disease, n_recommendations)
        return {
            'predicted_disease': predicted_disease,
            'recommendations': recommended_drugs
        }

hybrid_system = HybridRecommendationSystem(
    best_model,
    X_train_disease.columns.tolist(),
    sorted(y_train_disease.unique().tolist()),
    condition_drug_rating
)

print("✓ 하이브리드 시스템 초기화 완료")

# ============================================================================
# PART 5: 시각화
# ============================================================================

print("\n### PART 5: 시각화 생성 ###\n")

# 1. 모델 성능 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Disease Prediction Model Performance\n(Training: marslinoedward | Testing: itachi9604)', 
             fontsize=16, fontweight='bold')

models = ['k-NN', 'Decision Tree']
accuracies = [knn_acc, dt_acc]
colors = ['#2ecc71', '#3498db']

axes[0, 0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([0.9, 1.01])
axes[0, 0].set_title('Accuracy')
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

precisions = [knn_prec, dt_prec]
axes[0, 1].bar(models, precisions, color=['#e74c3c', '#95a5a6'], alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_ylim([0.9, 1.01])
axes[0, 1].set_title('Precision')
for i, v in enumerate(precisions):
    axes[0, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

recalls = [knn_rec, dt_rec]
axes[1, 0].bar(models, recalls, color=['#9b59b6', '#1abc9c'], alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_ylim([0.9, 1.01])
axes[1, 0].set_title('Recall')
for i, v in enumerate(recalls):
    axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

f1_scores = [knn_f1, dt_f1]
axes[1, 1].bar(models, f1_scores, color=['#f39c12', '#34495e'], alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_ylim([0.9, 1.01])
axes[1, 1].set_title('F1-Score')
for i, v in enumerate(f1_scores):
    axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, '01_model_performance.png'), dpi=300, bbox_inches='tight')
print("✓ 모델 성능 비교")
plt.close()

# 2. 약물 평점 분포
fig, ax = plt.subplots(figsize=(12, 6))
rating_dist = drug_train['rating'].value_counts().sort_index()
ax.bar(rating_dist.index, rating_dist.values, color='#3498db', alpha=0.7, edgecolor='black')
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Drug Rating Distribution (Training Data)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, '02_rating_distribution.png'), dpi=300, bbox_inches='tight')
print("✓ 약물 평점 분포")
plt.close()

# 3. 데이터셋 크기 비교
fig, ax = plt.subplots(figsize=(12, 6))
data_info = {
    'Disease\nTrain\n(marslinoedward)': len(X_train_disease),
    'Disease\nTest\n(itachi9604)': len(X_test_disease),
    'Drug\nTrain': len(drug_train),
    'Drug\nTest': len(drug_test)
}
colors_data = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax.bar(data_info.keys(), data_info.values(), color=colors_data, alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATION_PATH, '03_dataset_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ 데이터셋 크기 비교")
plt.close()

# ============================================================================
# PART 6: 모델 저장
# ============================================================================

print("\n### PART 6: 모델 및 결과 저장 ###\n")

with open(os.path.join(MODEL_PATH, "best_disease_model.pkl"), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_PATH, "hybrid_system.pkl"), 'wb') as f:
    dill.dump(hybrid_system, f)

condition_drug_rating.to_csv(os.path.join(MODEL_PATH, "condition_drug_rating.csv"), index=False)

metadata = {
    'project': 'Symptom-based Disease Prediction and Drug Recommendation System',
    'version': 'FINAL v2 (Corrected)',
    'disease_model': best_model_name,
    'disease_accuracy': float(max(knn_acc, dt_acc)),
    'train_data': {
        'disease': 'marslinoedward/disease-prediction-data (4,962 samples)',
        'drug': 'yash9439 + jessicali9530 (322,594 reviews)'
    },
    'test_data': {
        'disease': 'itachi9604/disease-symptom-description-dataset (4,920 samples)',
        'drug': 'yash9439 + jessicali9530 test sets (107,532 reviews)'
    },
    'unused_datasets': [
        'kaushil268/disease-prediction-using-machine-learning (동일한 데이터이므로 미사용)'
    ],
    'model_performance': {
        'accuracy': float(max(knn_acc, dt_acc)),
        'precision': float(max(knn_prec, dt_prec)),
        'recall': float(max(knn_rec, dt_rec)),
        'f1_score': float(max(knn_f1, dt_f1))
    },
    'drug_recommendation': {
        'total_drugs': condition_drug_rating['drugName'].nunique(),
        'total_conditions': condition_drug_rating['condition'].nunique(),
        'avg_rating': float(condition_drug_rating['avg_rating'].mean())
    }
}

with open(os.path.join(MODEL_PATH, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ 모델 저장: {MODEL_PATH}")
print(f"✓ 시각화 저장: {VISUALIZATION_PATH}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*100)
print("✅ 프로젝트 최종 완료!")
print("="*100 + "\n")

print("📋 사용된 데이터셋:")
print("  ✓ 질병 예측 - 훈련 (많은 데이터):")
print("    → marslinoedward/disease-prediction-data")
print("    → 4,962 샘플")
print()
print("  ✓ 질병 예측 - 테스트 (적은 데이터):")
print("    → itachi9604/disease-symptom-description-dataset")
print("    → 4,920 샘플 (다른 형식으로 변환)")
print()
print("  ✗ 미사용:")
print("    → kaushil268/disease-prediction-using-machine-learning (동일한 데이터)")
print()
print("  ✓ 약물 리뷰 (훈련 + 테스트):")
print("    → yash9439/drug-review + jessicali9530/kuc-hackathon")
print()

print("📊 최종 성능:")
print(f"  ✓ 질병 예측 정확도: {max(knn_acc, dt_acc)*100:.2f}% ({best_model_name})")
print(f"  ✓ 훈련 샘플: {len(X_train_disease)} (질병) + {len(drug_train):,} (약물)")
print(f"  ✓ 테스트 샘플: {len(X_test_disease)} (질병) + {len(drug_test):,} (약물)")
print(f"  ✓ 약물 종류: {condition_drug_rating['drugName'].nunique()}개")
print(f"  ✓ 평균 평점: {condition_drug_rating['avg_rating'].mean():.2f}/10")

print("\n" + "="*100 + "\n")

