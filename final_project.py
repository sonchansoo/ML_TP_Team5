"""
════════════════════════════════════════════════════════════════════════════════════════════════════
증상 기반 질병 예측 및 약물 추천 하이브리드 시스템 (v3 - 현실적 데이터셋 적용)

데이터셋 정보
────────────────────────────────────────────────────────────────────────────────────────────────────
[수정] 질병 예측:
  - aniruddha2000/symptom-based-disease-classification-dataset (25,200개)
  - 100% 정확도 문제를 해결하기 위해 실제 환자 증상과 유사한 비중복 데이터셋으로 교체
  
약물 추천:
  - yash9439/drug-review + jessicali9530 (데이터셋 동일)
════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

# 작업 디렉토리를 스크립트 위치로 설정 (numpy import 오류 방지)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # 작업 디렉토리를 스크립트 위치로 변경
    os.chdir(script_dir)
except NameError:
    # __file__이 없는 환경(예: Jupyter notebook)에서는 현재 디렉토리 사용
    pass

import pandas as pd
import numpy as np
import kagglehub
import pickle
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 분류 모델 Import
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 2. 전처리 및 평가 도구 Import
from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# 3. 기타 유틸리티 Import
import dill

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# 설정
# ============================================================================

MODEL_PATH = "models_final"
VISUALIZATION_PATH = "visualizations"
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

print("\n" + "="*100)
print("증상 기반 질병 예측 및 약물 추천 시스템")
print("="*100)

# ============================================================================
# PART 1: 데이터 로드
# ============================================================================

print("\n### PART 1: 데이터 로드 ###\n")

# 🔽 [수정] 새로운 'SympScan' 데이터셋 로드
print("📥 1️⃣  질병 예측 - 훈련 데이터 (SympScan)")
try:
    # API 다운로드 시도
    disease_path = kagglehub.dataset_download("behzadhassan/sympscan-symptomps-to-disease")
    disease_df = pd.read_csv(os.path.join(disease_path, "Diseases_and_Symptoms_dataset.csv"))
except Exception as e:
    print(f"  > API 다운로드 실패. 로컬 파일 로드 시도: 'Diseases_and_Symptoms_dataset.csv'")
    try:
        # 로컬 파일 로드 (이것을 사용합니다)
        disease_df = pd.read_csv("Diseases_and_Symptoms_dataset.csv")
    except FileNotFoundError:
        print("="*50)
        print("오류: 'Diseases_and_Symptoms_dataset.csv' 파일을 찾을 수 없습니다.")
        print("Kaggle에서 파일을 다운로드한 후, .py 파일과 같은 폴더에 넣어주세요.")
        print("="*50)
        exit() # 파일이 없으면 스크립트 중지

print(f"  ✓ 원본 로드: {disease_df.shape[0]} 샘플")
print(f"  ✓ 질병 종류: {disease_df['diseases'].nunique()}개")

X_df = disease_df.drop('diseases', axis=1) # 'disease' 컬럼을 제외한 모든 것이 증상(X)
y = disease_df['diseases'] # 'disease' 컬럼이 타겟(y)

print(f"  ✓ 총 특성(증상) 수: {X_df.shape[1]}개")

# 5. 최종 데이터셋을 훈련/테스트용으로 분할
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

# 하이브리드 시스템(PART 4)에서 사용할 증상 이름 리스트
all_feature_names = X_df.columns.tolist()

print(f"  ✓ 훈련 (80%): {len(X_train_disease)} 샘플")
print(f"  ✓ 테스트 (20%): {len(X_test_disease)} 샘플")
# -------------------------------------------------------------

print("\n📥 2️⃣  약물 리뷰 데이터 (yash9439 + jessicali9530)")
try:
    drug_path1 = kagglehub.dataset_download("yash9439/drug-review")
    drug_train1 = pd.read_csv(os.path.join(drug_path1, "drugsComTrain_raw.tsv"), sep='\t')
    drug_test1 = pd.read_csv(os.path.join(drug_path1, "drugsComTest_raw.tsv"), sep='\t')
except Exception as e:
    print(f"⚠️  yash9439 데이터셋 다운로드 실패: {e}")
    print("대체 경로에서 로드 시도: 'drugsComTrain_raw.tsv', 'drugsComTest_raw.tsv'")
    drug_train1 = pd.read_csv("drugsComTrain_raw.tsv", sep='\t')
    drug_test1 = pd.read_csv("drugsComTest_raw.tsv", sep='\t')

try:
    drug_path2 = kagglehub.dataset_download("jessicali9530/kuc-hackathon-winter-2018")
    drug_train2 = pd.read_csv(os.path.join(drug_path2, "drugsComTrain_raw.csv"))
    drug_test2 = pd.read_csv(os.path.join(drug_path2, "drugsComTest_raw.csv"))
except Exception as e:
    print(f"⚠️  jessicali9530 데이터셋 다운로드 실패: {e}")
    print("대체 경로에서 로드 시도: 'drugsComTrain_raw.csv', 'drugsComTest_raw.csv'")
    drug_train2 = pd.read_csv("drugsComTrain_raw.csv")
    drug_test2 = pd.read_csv("drugsComTest_raw.csv")

drug_train1 = drug_train1.drop('Unnamed: 0', axis=1, errors='ignore')
drug_test1 = drug_test1.drop('Unnamed: 0', axis=1, errors='ignore')
drug_train2 = drug_train2.drop('uniqueID', axis=1, errors='ignore')
drug_test2 = drug_test2.drop('uniqueID', axis=1, errors='ignore')

drug_train_all = pd.concat([drug_train1, drug_train2], ignore_index=True)
drug_test_all = pd.concat([drug_test1, drug_test2], ignore_index=True)

drug_train = drug_train_all.dropna(subset=['condition']).copy()
drug_test = drug_test_all.dropna(subset=['condition']).copy()

drug_train['condition'] = drug_train['condition'].str.strip()
drug_train['drugName'] = drug_train['drugName'].str.strip()

print(f"   훈련: {len(drug_train):,} 리뷰")
print(f"   테스트: {len(drug_test):,} 리뷰")

# ============================================================================
# PART 2: 질병 예측 모델
# ============================================================================

print("\n### PART 2: 질병 예측 모델 구축 ###\n")
print("... (5개 모델 학습 및 평가 중) ...\n")

# --- 모델 리스트 ---
models = {
    "k-NN": KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM (선형)": SVC(kernel='linear', probability=True, random_state=42)
}

results = {}
model_objects = {}

# --- 학습 및 평가 루프 ---
for name, model in models.items():
    print(f"⏳ {name} 모델 학습 중...")
    model.fit(X_train_disease, y_train_disease)
    y_pred = model.predict(X_test_disease)
    
    acc = accuracy_score(y_test_disease, y_pred)
    prec = precision_score(y_test_disease, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_disease, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_disease, y_pred, average='weighted', zero_division=0)
    
    print(f"📊 {name} 성능 (aniruddha2000 분할 데이터):")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
    
    results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    model_objects[name] = model

# --- 최고 성능 모델 선택 로직 ---
best_model_name = max(results, key=lambda k: results[k]['acc'])
best_acc = results[best_model_name]['acc']
best_model = model_objects[best_model_name]

print(f"✓ 최고 성능 모델: {best_model_name} (정확도: {best_acc:.4f})")

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
    
    [수정]
    - __init__: 리뷰 텍스트 원본(drug_train)을 시스템에 전달
    - recommend_drugs_memory: (기존) 평점/리뷰 수 기반 (메모리 기반)
    - recommend_drugs_content: (신규) 리뷰 텍스트 키워드 분석 (콘텐츠 기반)
    """
    
    def __init__(self, disease_model, feature_names, target_names, 
                 condition_drug_rating, drug_review_df): # drug_review_df (원본 리뷰)
        
        self.disease_model = disease_model
        self.feature_names = feature_names
        self.target_names = target_names
        self.condition_drug_rating = condition_drug_rating # 메모리 기반용
        self.drug_review_df = drug_review_df # 콘텐츠 기반용
        
        self.model_feature_names = self.disease_model.feature_names_in_

        # 콘텐츠 분석용 키워드
        self.positive_keywords = [
            'effective', 'great', 'worked well', 'no side effects', 'fast', 
            'amazing', 'helpful', 'saved', 'life saver', 'best'
        ]
        self.negative_keywords = [
            'ineffective', 'terrible', 'horrible', 'worst', 'side effects', 
            'no help', 'awful', 'pain', 'did not work'
        ]
    
    def predict_disease(self, symptoms_dict):
        symptoms_vector = pd.Series(0, index=self.feature_names)
        valid_symptoms = {k: v for k, v in symptoms_dict.items() if k in self.feature_names}
        symptoms_vector.update(pd.Series(valid_symptoms))
        symptoms_vector_aligned = symptoms_vector[self.model_feature_names].values.reshape(1, -1)
        return self.disease_model.predict(symptoms_vector_aligned)[0]
    
    # ----------------------------------------------------------------------
    # 1. 메모리 기반 추천
    # ----------------------------------------------------------------------
    def recommend_drugs_memory(self, disease, n_recommendations=5):
        """
        (기존 로직) 2단계: 질병 → 약물 추천 (메모리 기반 - 평점/리뷰 수)
        """
        disease_drugs = self.condition_drug_rating[
            self.condition_drug_rating['condition'].str.lower() == disease.lower() # 소문자 비교
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

    # ----------------------------------------------------------------------
    # 2. 콘텐츠 기반 추천
    # ----------------------------------------------------------------------
    def recommend_drugs_content(self, disease, n_recommendations=5):
        """
        (신규 로직) 2단계: 질병 → 약물 추천 (콘텐츠 기반 - 리뷰 텍스트 분석)
        """
        # 1. 원본 리뷰DF에서 해당 질병 리뷰만 필터링 (리뷰 텍스트가 필요함)
        disease_reviews = self.drug_review_df[
            self.drug_review_df['condition'].str.lower() == disease.lower()
        ].copy()
        
        if disease_reviews.empty:
            return pd.DataFrame(columns=['drugName', 'content_score', 'total_reviews'])

        # 2. 리뷰 텍스트를 소문자로 변환 (결측치 처리)
        disease_reviews['review_text'] = disease_reviews['review'].fillna('').astype(str).str.lower()
        
        # 3. 약물별로 키워드 점수 계산
        def calculate_content_score(reviews_text):
            score = 0
            text = " ".join(reviews_text) # 한 약물의 모든 리뷰를 합침
            for kw in self.positive_keywords:
                score += text.count(kw)
            for kw in self.negative_keywords:
                score -= text.count(kw)
            return score

        # 4. 'drugName'으로 그룹화하여 리뷰 수(count)와 콘텐츠 점수(score) 계산
        drug_content_scores = disease_reviews.groupby('drugName')['review_text'].agg(
            total_reviews='count',
            content_score=calculate_content_score
        ).reset_index()

        # 5. 콘텐츠 점수 > 리뷰 수 순으로 정렬
        return drug_content_scores.sort_values(
            by=['content_score', 'total_reviews'], 
            ascending=[False, False]
        ).head(n_recommendations)
    # ----------------------------------------------------------------------

    def predict_and_recommend(self, symptoms_dict, n_recommendations=5, method='memory'):
        """
        [수정] End-to-End: 증상 → 질병 → 약물 추천
        method: 'memory' (평점) 또는 'content' (리뷰 텍스트) 중 선택
        """
        predicted_disease = self.predict_disease(symptoms_dict)
        
        if method == 'content':
            recommendations = self.recommend_drugs_content(predicted_disease, n_recommendations)
            rec_type = "Content-Based"
        else: # 기본값 'memory'
            recommendations = self.recommend_drugs_memory(predicted_disease, n_recommendations)
            rec_type = "Memory-Based (Ratings)"
            
        return {
            'predicted_disease': predicted_disease,
            'recommendation_type': rec_type,
            'recommendations': recommendations
        }

# 시스템 초기화 시 'drug_train' 원본을 전달합니다.
hybrid_system = HybridRecommendationSystem(
    best_model,
    all_feature_names, 
    sorted(y_train_disease.unique().tolist()),
    condition_drug_rating,
    drug_train # PART 1에서 로드한 약물 리뷰 원본 DF
)

print("✓ 하이브리드 시스템 초기화 완료 (메모리 + 콘텐츠 기반 추천기 탑재)")

# ============================================================================
# PART 5: 시각화
# ============================================================================

print("\n### PART 5: 시각화 생성 ###\n")

# 1. 모델 성능 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# 차트 제목
fig.suptitle('Disease Prediction Model Performance\n(Dataset: aniruddha2000, 80/20 Split)', 
             fontsize=16, fontweight='bold')

# 5개 모델 결과 반영
model_names = list(results.keys())
accuracies = [results[name]['acc'] for name in model_names]
precisions = [results[name]['prec'] for name in model_names]
recalls = [results[name]['rec'] for name in model_names]
f1_scores = [results[name]['f1'] for name in model_names]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6', '#1abc9c']

# Accuracy
axes[0, 0].bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.02])
axes[0, 0].set_title('Accuracy')
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# Precision
axes[0, 1].bar(model_names, precisions, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_ylim([min(precisions) - 0.05, max(precisions) + 0.02])
axes[0, 1].set_title('Precision')
for i, v in enumerate(precisions):
    axes[0, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# Recall
axes[1, 0].bar(model_names, recalls, color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_ylim([min(recalls) - 0.05, max(recalls) + 0.02])
axes[1, 0].set_title('Recall')
for i, v in enumerate(recalls):
    axes[1, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# F1-Score
axes[1, 1].bar(model_names, f1_scores, color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_ylim([min(f1_scores) - 0.05, max(f1_scores) + 0.02])
axes[1, 1].set_title('F1-Score')
for i, v in enumerate(f1_scores):
    axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

plt.setp(axes[0, 0].get_xticklabels(), rotation=15, ha='right')
plt.setp(axes[0, 1].get_xticklabels(), rotation=15, ha='right')
plt.setp(axes[1, 0].get_xticklabels(), rotation=15, ha='right')
plt.setp(axes[1, 1].get_xticklabels(), rotation=15, ha='right')
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
    'Disease\nTrain\n(aniruddha2000)': len(X_train_disease),
    'Disease\nTest\n(aniruddha2000)': len(X_test_disease),
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

# 메타데이터 업데이트
metadata = {
    'project': 'Symptom-based Disease Prediction and Drug Recommendation System',
    'version': 'v3 (Realistic Dataset)',
    'disease_model': best_model_name,
    'disease_accuracy': float(best_acc),
    'train_data': {
        'disease': f'aniruddha2000/symptom-based-disease-classification-dataset ({len(X_train_disease)} samples)',
        'drug': f'yash9439 + jessicali9530 ({len(drug_train):,} reviews)'
    },
    'test_data': {
        'disease': f'aniruddha2000/symptom-based-disease-classification-dataset ({len(X_test_disease)} samples)',
        'drug': f'yash9439 + jessicali9530 test sets ({len(drug_test):,} reviews)'
    },
    'model_performance': results[best_model_name],
    'drug_recommendation': {
        'total_drugs': condition_drug_rating['drugName'].nunique(),
        'total_conditions': condition_drug_rating['condition'].nunique(),
        'avg_rating': float(condition_drug_rating['avg_rating'].mean())
    }
}

with open(os.path.join(MODEL_PATH, "metadata.json"), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f" 모델 저장: {MODEL_PATH}")
print(f" 시각화 저장: {VISUALIZATION_PATH}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*100)
print(" 프로젝트 최종")
print("="*100 + "\n")


print(" 최종 성능:")
# 최고 모델의 실제 정확도 출력
print(f"   질병 예측 정확도: {best_acc*100:.2f}% ({best_model_name})")
print(f"   훈련 샘플: {len(X_train_disease)} (질병) + {len(drug_train):,} (약물)")
print(f"   테스트 샘플: {len(X_test_disease)} (질병) + {len(drug_test):,} (약물)")
print(f"   약물 종류: {condition_drug_rating['drugName'].nunique()}개")
print(f"   평균 평점: {condition_drug_rating['avg_rating'].mean():.2f}/10")

print("\n" + "="*100 + "\n")