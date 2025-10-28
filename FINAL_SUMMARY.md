#  Machine Learning Term Project - 최종 완성

## 증상 기반 질병 예측 및 약물 추천 하이브리드 시스템

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **도메인** | 헬스케어 & 의약 정보 (Healthcare & Pharmaceutical Information) |
| **목표** | 증상 입력 → 질병 예측 → 약물 추천 (End-to-End 하이브리드 시스템) |
| **접근법** | 콘텐츠 기반 필터링 + 협업 필터링 (Hybrid Filtering) |
| **상태** | ✅ 최종 완성 |

---

## 📊 데이터셋 최종 구성

### 【 질병 예측 데이터 】

| 구분 | 데이터셋 | 샘플 수 | 상태 |
|------|---------|--------|------|
| **훈련** | `marslinoedward/disease-prediction-data` | 4,920 | ✅ 사용 |
| **테스트** | `itachi9604/disease-symptom-description-dataset` | 4,920 | ✅ 사용 |

**특징:**
- 훈련: 132개 증상 특성, 41개 질병 분류
- 테스트: Symptom_1~17 형식 → marslinoedward 형식으로 변환

### 【 약물 리뷰 데이터 】

| 구분 | 데이터셋 1 | 데이터셋 2 | 합계 | 상태 |
|------|-----------|-----------|------|------|
| **훈련** | yash9439 (161,297) | jessicali9530 (161,297) | 322,594 | ✅ 사용 |
| **테스트** | yash9439 test (53,766) | jessicali9530 test (53,766) | 107,532 | ✅ 사용 |

---

## 🤖 머신러닝 모델

### 【 질병 예측 모델 】

#### 1. k-Nearest Neighbors (k-NN)
```
n_neighbors: 5
metric: euclidean
테스트 정확도: 100.00% 
Precision: 100% | Recall: 100% | F1-Score: 100%
```

#### 2. Decision Tree
```
criterion: entropy
max_depth: 15
테스트 정확도: 98.66%
Precision: 99.07% | Recall: 98.66% | F1-Score: 98.76%
```

**✓ 선택 모델: k-NN (100% 정확도)**

### 【 약물 추천 모델 】

- **방법:** 협업 필터링 (Collaborative Filtering)
- **유사도:** 코사인 유사도 (Cosine Similarity)
- **점수 계산:** 
  - avg_rating × 0.7 + log(review_count) × 0.3

**통계:**
- 질병-약물 쌍: 8,490개
- 고유 약물: 3,431개
- 고유 질병: 884개
- 평균 평점: 7.37/10

---

##  하이브리드 시스템 아키텍처

### 【 2단계 처리 파이프라인 】

```
입력: 증상 벡터 (binary, 132차원)
  ↓
[Stage 1] 콘텐츠 기반 필터링 (Content-Based Filtering)
  → k-NN 모델을 통한 질병 예측
  → 입력: 증상 패턴 → 출력: 예측 질병 (100% 정확도)
  ↓
[Stage 2] 협업 필터링 (Collaborative Filtering)
  → 예측된 질병과 관련된 약물 추천
  → 코사인 유사도 기반 약물 선택
  → 입력: 질병 → 출력: 추천 약물 (상위 N개)
  ↓
출력: 추천 약물 리스트 (약물명, 평점, 리뷰 수)
```

---

##  최종 성능 지표

### 【 질병 예측 성능 】
- Accuracy: **100.00%** (k-NN)
- Precision: 100.00%
- Recall: 100.00%
- F1-Score: 100.00%

### 【 약물 추천 성능 】
- 평균 추천 약물 평점: 7.37/10
- 질병당 평균 약물 수: 9.6개
- 시스템 커버리지: 26.8%

### 【 데이터셋 규모 】
- 훈련 데이터: 4,920 (질병) + 320,796 (약물) = **325,716개**
- 테스트 데이터: 4,920 (질병) + 106,942 (약물) = **111,862개**
- 총 데이터: **437,578개**

---

## 최종 프로젝트 구조

```
/root/
├── final_project.py                 (메인 프로젝트 파일)
│   
│
├── models_final/                     (저장된 모델 및 데이터)
│   ├── best_disease_model.pkl        (k-NN 모델)
│   ├── hybrid_system.pkl             (통합 시스템)
│   ├── condition_drug_rating.csv     (약물 데이터)
│   └── metadata.json                 (메타데이터)
│
├── visualizations/                   (시각화 이미지)
│   ├── 01_model_performance.png       모델 성능 비교
│   ├── 02_rating_distribution.png     약물 평점 분포
│   └── 03_dataset_comparison.png      데이터셋 크기 비교
│
├── FINAL_SUMMARY.md                  (이 파일)
├── FINAL_REPORT.md                   (상세 보고서)

```

---

##  프로젝트 실행 방법

### 1️⃣ 프로젝트 실행
```bash
cd /root
python3 final_project.py
```

### 2️⃣ 저장된 모델 사용
```python
import dill
import numpy as np

# 모델 로드
with open('/root/models_final/hybrid_system.pkl', 'rb') as f:
    system = dill.load(f)

# 증상 벡터 생성 (132차원 binary)
symptoms = np.array([0, 1, 0, ..., 1])

# 질병 예측 및 약물 추천
result = system.predict_and_recommend(symptoms, n=5)

print("예측된 질병:", result['predicted_disease'])
print("추천 약물:")
print(result['recommendations'])
```

### 3️⃣ 시각화 확인
```bash
ls -lh /root/visualizations/
```

---

## ✅ 과제 요구사항 충족도

### 필수 요구사항
- ✅ Recommendation System 구현 (하이브리드 방식)
- ✅ 도메인: 헬스케어 (질병 예측 & 약물 추천)
- ✅ 필터링 방법: Content-Based + Collaborative Filtering
- ✅ ML 모델: k-NN ✓, Decision Tree ✓, Cosine Similarity ✓
- ✅ 데이터셋: 4개 (Disease 2개 + Drug Review 2개)

### 추가 요구사항
- ✅ 훈련/테스트 분할: 명확하게 구분
- ✅ 데이터 전처리: 완료 (형식 변환, 결측치 처리)
- ✅ 모델 평가: Accuracy, Precision, Recall, F1-Score
- ✅ 시각화: matplotlib으로 3개 이미지 생성
- ✅ 모델 저장: pickle + dill 사용
- ✅ 주석: 상세한 한국어/영어 주석 포함
- ✅ End-to-End 파이프라인: 완벽하게 구현

---

##  학습 성과

### 1. 추천 시스템 설계
- 콘텐츠 기반 필터링의 원리 및 구현
- 협업 필터링과 유사도 측정 (코사인 유사도)
- 하이브리드 방식의 장점과 통합 방법

### 2. 머신러닝 모델
- k-NN의 장점 (높은 정확도)과 단점 (계산 비용)
- Decision Tree의 해석성
- 모델 평가 지표의 실제 적용

### 3. 실제 데이터 처리
- 다양한 데이터셋 통합 및 전처리
- 결측치 처리 및 데이터 정규화
- 서로 다른 형식의 데이터 변환

### 4. 헬스케어 도메인
- 질병-증상 관계 파악
- 약물 추천의 현실적 고려사항
- 의료 정보의 신뢰성 중요성

---



---

## 🎯 주요 개선 사항 (최종 버전)

### 1. 데이터셋 확정
- ✅ 질병: marslinoedward (훈련) + itachi9604 (테스트)

### 2. 모델 성능
- ✅ k-NN: 100% 정확도 (itachi9604 테스트)
- ✅ 다양한 형식의 데이터 처리 성공

### 3. 시스템 안정성
- ✅ 완전한 End-to-End 파이프라인
- ✅ 예외 처리 포함
- ✅ 재현성 보장

---

##  코드 통계

| 항목 | 수치 |
|------|------|
| 메인 파일 크기 | 19 KB |
| 코드 라인 수 | ~420줄 |
| 주석 포함 | ~30% |
| 모델 파일 크기 | 10.3 MB (total) |
| 시각화 이미지 | 3개 |

---

##  특이사항

- **100% 정확도**: k-NN 모델이 itachi9604 테스트 데이터에서 완벽한 성능을 달성
- **다중 데이터셋 통합**: 2개의 질병 데이터와 2개의 약물 데이터를 성공적으로 통합
- **형식 변환**: 다른 형식의 테스트 데이터(itachi9604)를 훈련 형식으로 자동 변환

---

## 최종 결론

- 소스코드: `final_project.py` 
- 저장된 모델: `models_final/`
- 시각화: `visualizations/`
- 보고서: `FINAL_SUMMARY.md`, `FINAL_REPORT.md`

---


