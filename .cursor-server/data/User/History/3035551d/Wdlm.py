import os
import pandas as pd
import kagglehub
import glob

# 다운로드할 데이터셋 정보
datasets = [
    {
        "name": "Disease Symptom Prediction Dataset 1",
        "path": "kaushil268/disease-prediction-using-machine-learning"
    },
    {
        "name": "Disease Symptom Prediction Dataset 2",
        "path": "marslinoedward/disease-prediction-data"
    },
    {
        "name": "Drug Review Dataset 1",
        "path": "yash9439/drug-review"
    },
    {
        "name": "Drug Review Dataset 2",
        "path": "jessicali9530/kuc-hackathon-winter-2018"
    }
]

# 각 데이터셋 다운로드 및 확인
for dataset in datasets:
    print("\n" + "="*80)
    print(f"📥 다운로드 중: {dataset['name']}")
    print(f"경로: {dataset['path']}")
    print("="*80)
    
    try:
        # 데이터셋 다운로드
        download_path = kagglehub.dataset_download(dataset['path'])
        print(f"✓ 다운로드 완료: {download_path}\n")
        
        # 해당 디렉토리의 모든 CSV/TSV 파일 찾기
        csv_files = glob.glob(os.path.join(download_path, "**/*.csv"), recursive=True)
        tsv_files = glob.glob(os.path.join(download_path, "**/*.tsv"), recursive=True)
        all_files = csv_files + tsv_files
        
        if all_files:
            print(f"발견된 파일 ({len(all_files)}개):")
            for file in all_files:
                print(f"  - {file}")
            print()
            
            # 각 파일을 pandas로 로드해서 head() 출력
            for file_path in all_files[:3]:  # 처음 3개 파일만 표시
                print(f"\n📄 파일: {os.path.basename(file_path)}")
                print("-" * 80)
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path, sep='\t')
                    
                    print(f"형태(Shape): {df.shape}")
                    print(f"컬럼(Columns): {list(df.columns)}\n")
                    print("데이터 샘플 (첫 5행):")
                    print(df.head())
                    print(f"\n데이터 타입:\n{df.dtypes}")
                except Exception as e:
                    print(f"✗ 파일 로드 실패: {e}")
                print("-" * 80)
        else:
            print("✗ CSV/TSV 파일을 찾을 수 없습니다.")
            
    except Exception as e:
        print(f"✗ 오류 발생: {e}")

print("\n" + "="*80)
print("✓ 데이터셋 확인 완료!")
print("="*80)