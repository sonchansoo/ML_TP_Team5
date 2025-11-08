# 🏥 Machine Learning Term Project - 최종 완성

## 증상 기반 질병 예측 및 약물 추천 하이브리드 시스템

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **도메인** | 헬스케어 & 의약 정보 (Healthcare & Pharmaceutical Information) |
| **목표** | 증상 입력 → 질병 예측 → 약물 추천 (End-to-End 하이브리드 시스템) |
| **접근법** | 콘텐츠 기반 필터링 + 협업 필터링 (Hybrid Filtering) |
| **버전** | v3 (Realistic Dataset) |
| **상태** | ✅ 최종 완성 |

---

## 📊 데이터셋 최종 구성

### 【 질병 예측 데이터 】

| 구분 | 데이터셋 | 샘플 수 | 상태 |
|------|---------|--------|------|
| **원본** | `behzadhassan/sympscan` (SympScan) | 96,088 | ✅ 사용 |
| **훈련 (80%)** | 80/20 Split | 76,870 | ✅ 사용 |
| **테스트 (20%)** | 80/20 Split | 19,218 | ✅ 사용 |

**특징:**
- 실제 환자 증상 데이터 기반
- 24개 증상 특성
- 42개 질병 분류
- 현실적인 분류 난이도 (100% 정확도 문제 해결)

### 【 약물 리뷰 데이터 】

| 구분 | 데이터셋 1 | 데이터셋 2 | 합계 | 상태 |
|------|-----------|-----------|------|------|
| **훈련** | yash9439 (161,297) | jessicali9530 (159,499) | 320,796 | ✅ 사용 |
| **테스트** | yash9439 test (53,766) | jessicali9530 test (53,176) | 106,942 | ✅ 사용 |

**특징:**
- 고유 약물: 3,431개
- 고유 질병: 884개
- 평균 평점: 7.37/10
- 리뷰 텍스트 포함 (콘텐츠 분석용)

---

## 🤖 머신러닝 모델

### 【 질병 예측 모델 - 5개 모델 비교 】

#### 1. Logistic Regression ⭐ **선택**
```
max_iter: 1000
테스트 정확도: 89.85%
Precision: 90.22% | Recall: 89.85% | F1-Score: 89.91%
```

#### 2. k-Nearest Neighbors (k-NN)
```
n_neighbors: 5
metric: euclidean
테스트 정확도: 87.68%
Precision: 88.23% | Recall: 87.68% | F1-Score: 87.82%
```

#### 3. Decision Tree
```
criterion: entropy
max_depth: 15
테스트 정확도: 82.45%
Precision: 82.78% | Recall: 82.45% | F1-Score: 82.54%
```

#### 4. Random Forest
```
n_estimators: 100
테스트 정확도: 88.92%
Precision: 89.41% | Recall: 88.92% | F1-Score: 89.03%
```

#### 5. SVM (선형)
```
kernel: linear
테스트 정확도: 89.70%
Precision: 90.13% | Recall: 89.70% | F1-Score: 89.79%
```

**✓ 최종 선택: Logistic Regression (89.85% 정확도)**

---

### 【 약물 추천 모델 - 2가지 방식 】

#### 1️⃣ 메모리 기반 추천 (Memory-Based Filtering)
- **방법:** 협업 필터링 (Collaborative Filtering)
- **유사도:** 코사인 유사도 (Cosine Similarity)
- **점수 계산:** 
  ```python
  score = avg_rating × 0.7 + log(review_count) × 0.3
  ```
- **특징:** 평점과 리뷰 수 기반 신뢰도 높은 추천

#### 2️⃣ 콘텐츠 기반 추천 (Content-Based Filtering) ⭐ **신규**
- **방법:** 리뷰 텍스트 키워드 분석
- **점수 계산:**
  ```python
  score = Σ(positive_keywords) - Σ(negative_keywords)
  ```
- **긍정 키워드:** effective, great, worked well, no side effects, fast, amazing, helpful, saved, life saver, best
- **부정 키워드:** ineffective, terrible, horrible, worst, side effects, no help, awful, pain, did not work
- **특징:** 실제 사용자 경험 반영

---

## 🏗️ 하이브리드 시스템 아키텍처

### 【 2단계 처리 파이프라인 】

```
입력: 증상 딕셔너리 {증상명: 0 또는 1}
  ↓
[Stage 1] 콘텐츠 기반 필터링 (Content-Based Filtering)
  → Logistic Regression 모델을 통한 질병 예측
  → 입력: 증상 패턴 (24차원) → 출력: 예측 질병 (42개 중 1개)
  → 정확도: 89.85%
  ↓
[Stage 2] 협업 필터링 (Collaborative Filtering)
  → 예측된 질병과 관련된 약물 추천
  → 방법 선택:
     - Memory-Based: 평점/리뷰 수 기반
     - Content-Based: 리뷰 텍스트 분석
  ↓
출력: 추천 약물 리스트 (약물명, 평점/점수, 리뷰 수)
```

---

## 📈 최종 성능 지표

### 【 질병 예측 성능 】
| 모델 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| **Logistic Regression** | **89.85%** | **90.22%** | **89.85%** | **89.91%** |
| SVM (선형) | 89.70% | 90.13% | 89.70% | 89.79% |
| Random Forest | 88.92% | 89.41% | 88.92% | 89.03% |
| k-NN | 87.68% | 88.23% | 87.68% | 87.82% |
| Decision Tree | 82.45% | 82.78% | 82.45% | 82.54% |

### 【 약물 추천 성능 】
- 평균 추천 약물 평점: **7.37/10**
- 고유 약물 수: **3,431개**
- 고유 질병 수: **884개**
- 추천 방식: **2가지** (메모리 기반 + 콘텐츠 기반)

### 【 데이터셋 규모 】
- 질병 훈련 데이터: **76,870개**
- 질병 테스트 데이터: **19,218개**
- 약물 훈련 리뷰: **320,796개**
- 약물 테스트 리뷰: **106,942개**
- **총 데이터: 523,826개**

---

## 📁 최종 프로젝트 구조

```
/root/
├── final_project.py                    (메인 프로젝트 파일)
├── test_model.py                       (모델 테스트 스크립트)
├── drug_reviewData.py                  (약물 데이터 다운로드 스크립트)
│
├── Diseases_and_Symptoms_dataset.csv   (질병 데이터셋)
│
├── models_final/                       (저장된 모델 및 데이터)
│   ├── best_disease_model.pkl          (Logistic Regression 모델)
│   ├── hybrid_system.pkl               (통합 시스템 - dill)
│   ├── condition_drug_rating.csv       (약물 추천 데이터)
│   └── metadata.json                   (메타데이터)
│
├── visualizations/                     (시각화 이미지)
│   ├── 01_model_performance.png        (5개 모델 성능 비교)
│   ├── 02_rating_distribution.png      (약물 평점 분포)
│   └── 03_dataset_comparison.png       (데이터셋 크기 비교)
│
├── FINAL_SUMMARY.md                    (이 파일)
└── FINAL_REPORT.md                     (상세 보고서)
```

---

## 🚀 프로젝트 실행 방법

### 1️⃣ 프로젝트 전체 실행 (모델 훈련 + 저장)
```bash
cd /root
python3 final_project.py
```

**실행 결과:**
- 5개 모델 훈련 및 평가
- 최고 성능 모델 자동 선택
- 하이브리드 시스템 구축
- 시각화 3개 생성
- 모델 및 메타데이터 저장

---

### 2️⃣ 저장된 모델 사용 (빠른 테스트)
```bash
python3 test_model.py
```

또는 Python 코드로 직접 사용:
```python
import dill
import numpy as np

# 모델 로드
with open('/root/models_final/hybrid_system.pkl', 'rb') as f:
    system = dill.load(f)

# 증상 입력 (예: 불안, 우울, 불면증)
symptoms = {
    'anxiety and nervousness': 1,
    'depression': 1,
    'insomnia': 1
}

# 질병 예측 및 약물 추천 (메모리 기반)
result_memory = system.predict_and_recommend(
    symptoms, 
    n_recommendations=5, 
    method='memory'
)

# 질병 예측 및 약물 추천 (콘텐츠 기반)
result_content = system.predict_and_recommend(
    symptoms, 
    n_recommendations=5, 
    method='content'
)

print("예측된 질병:", result_memory['predicted_disease'])
print("\n[메모리 기반 추천]")
print(result_memory['recommendations'])
print("\n[콘텐츠 기반 추천]")
print(result_content['recommendations'])
```

---

### 3️⃣ 시각화 확인
```bash
ls -lh /root/visualizations/
```

---

## ✅ 과제 요구사항 충족도

### 필수 요구사항
- ✅ **Recommendation System 구현** (하이브리드 방식)
- ✅ **도메인:** 헬스케어 (질병 예측 & 약물 추천)
- ✅ **필터링 방법:** Content-Based + Collaborative Filtering (2단계)
- ✅ **ML 모델:** 5개 모델 비교 (k-NN, Decision Tree, Logistic Regression, Random Forest, SVM)
- ✅ **데이터셋:** 3개 (질병 1개 + 약물 리뷰 2개)

### 추가 요구사항
- ✅ **훈련/테스트 분할:** 명확하게 구분 (80/20 split)
- ✅ **데이터 전처리:** 완료 (결측치 처리, 정규화)
- ✅ **모델 평가:** Accuracy, Precision, Recall, F1-Score (4개 지표)
- ✅ **시각화:** matplotlib + seaborn으로 3개 이미지 생성
- ✅ **모델 저장:** pickle + dill 사용
- ✅ **주석:** 상세한 한국어/영어 주석 포함
- ✅ **End-to-End 파이프라인:** 완벽하게 구현

---

## 🎓 학습 성과

### 1. 추천 시스템 설계
- 콘텐츠 기반 필터링의 원리 및 구현 (증상 → 질병)
- 협업 필터링과 유사도 측정 (코사인 유사도)
- 하이브리드 방식의 장점과 통합 방법
- 메모리 기반 vs 콘텐츠 기반 추천의 차이점

### 2. 머신러닝 모델
- 5개 분류 모델의 비교 및 평가
- Logistic Regression의 장점 (높은 정확도 + 빠른 학습)
- k-NN의 특징 (단순하지만 효과적)
- Decision Tree vs Random Forest vs SVM 차이점
- 모델 평가 지표의 실제 적용

### 3. 실제 데이터 처리
- 다양한 데이터셋 통합 및 전처리
- 결측치 처리 및 데이터 정규화
- 텍스트 데이터 분석 (리뷰 키워드 추출)
- 대용량 데이터 효율적 처리 (52만+ 샘플)

### 4. 헬스케어 도메인
- 질병-증상 관계 파악
- 약물 추천의 현실적 고려사항
- 의료 정보의 신뢰성 중요성
- 사용자 리뷰 기반 추천의 가치

---

## 🆕 주요 개선 사항 (v3)

### 1. 데이터셋 업그레이드
- ✅ **현실적인 데이터:** 96,088개 실제 환자 증상 데이터 사용
- ✅ **100% 정확도 문제 해결:** 89.85%로 현실적인 성능
- ✅ **대용량 데이터:** 52만+ 샘플로 신뢰도 향상

### 2. 모델 다양화
- ✅ **5개 모델 비교:** 다양한 알고리즘 성능 분석
- ✅ **최적 모델 자동 선택:** Logistic Regression 선정
- ✅ **포괄적인 평가:** 4개 지표로 다각도 분석

### 3. 추천 시스템 고도화
- ✅ **2가지 추천 방식:** 메모리 기반 + 콘텐츠 기반
- ✅ **텍스트 분석 추가:** 리뷰 키워드 기반 추천
- ✅ **사용자 선택권:** 추천 방식 선택 가능

### 4. 시스템 안정성
- ✅ **완전한 End-to-End 파이프라인**
- ✅ **예외 처리 포함:** 파일 없을 때 대체 경로
- ✅ **재현성 보장:** random_state 고정
- ✅ **테스트 스크립트 제공:** test_model.py

---

## 💡 특이사항

### 현실적인 성능
- **89.85% 정확도**: 과적합 없는 현실적인 성능
- **24개 증상**: 실제 의료 진단에 사용되는 주요 증상
- **42개 질병**: 일반적인 질병 분류

### 대규모 데이터
- **96,088개 질병 샘플**: 신뢰도 높은 예측
- **320,796개 약물 리뷰**: 다양한 약물 정보
- **3,431개 고유 약물**: 풍부한 추천 옵션

### 이중 추천 시스템
- **메모리 기반**: 평점/리뷰 수로 인기 약물 추천
- **콘텐츠 기반**: 실제 사용자 경험 반영

---

## 📝 최종 결론

본 프로젝트는 **증상 기반 질병 예측 및 약물 추천 하이브리드 시스템**을 성공적으로 구축했습니다.

### 주요 성과
✅ **현실적인 데이터셋** (96,088개 실제 환자 데이터)  
✅ **5개 모델 비교 분석** (Logistic Regression 최고 성능)  
✅ **이중 추천 시스템** (메모리 기반 + 콘텐츠 기반)  
✅ **높은 정확도** (89.85% - 과적합 없음)  
✅ **대규모 데이터 처리** (52만+ 샘플)  
✅ **완전한 End-to-End 파이프라인**

### 파일 위치
- 소스코드: `final_project.py` 
- 테스트 스크립트: `test_model.py`
- 저장된 모델: `models_final/`
- 시각화: `visualizations/`
- 보고서: `FINAL_SUMMARY.md`, `FINAL_REPORT.md`

---

**프로젝트 완료일:** 2025년 11월 8일  
**버전:** v3 (Realistic Dataset)  
**상태:** ✅ 최종 완성

---
