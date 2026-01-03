# Level Crossing Obstacle Prediction Pipeline

이 프로젝트는 **연속 2프레임(2초 간격) + rail segmentation mask**를 입력으로 받아, **철로 위 정지 차량(최대 3대)**의 존재 확률과 bounding box 좌표를 예측하여 `submission.csv`로 저장합니다.

## 1. Data Overview

### 1.1 Image Files

각 샘플은 아래 3개 파일로 구성됩니다.

* `####_1.jpg` : 첫 번째 프레임
* `####_2.jpg` : 2초 뒤 프레임
* `####_rail.png` : rail segmentation mask (철로 영역 = 255, 배경 = 0)

> 파일명은 `0001`, `0003`처럼 **4자리 zero-padding**이며, **ID가 연속적이지 않을 수 있습니다.**
> 따라서 `range()`로 고정된 id를 생성하지 않고, 폴더를 스캔하여 존재하는 ID만 사용합니다.

### 1.2 CSV Files

#### `train.csv` columns

* Meta columns: `ID`, `image-3sec`, `image`, `segmentedRailImage`
* Target columns (15):
  `probaObstacle1, x1, dx1, y1, dy1, …, probaObstacle3, x3, dx3, y3, dy3`

#### `test.csv` columns

* Meta columns only: `ID`, `image-3sec`, `image`, `segmentedRailImage`
* No targets

> `image-3sec`, `image`, `segmentedRailImage`는 **파일명/경로 문자열**이므로, 모델 입력으로 직접 사용하지 않고 실제 이미지로부터 수치 feature를 추출합니다.

---

## 2. Overall Pipeline

전체 흐름은 아래 4단계로 구성됩니다.

1. **Valid ID scanning**
2. **Feature extraction from images** (KLT + frame differencing 기반)
3. **Merge extracted features with train targets**
4. **Model training → test inference → submission.csv 저장**

---

## 3. Step-by-Step Pipeline Details

### Step 1) Valid ID Scanning

데이터 폴더를 스캔하여 `####_1.jpg` 패턴을 가진 ID를 수집하고, `####_2.jpg`, `####_rail.png`까지 모두 존재하는 ID만 사용합니다.

* 이유: 중간 ID가 빠져 있을 수 있어 `range()` 사용 시 `FileNotFoundError` 발생 가능

산출물:

* `train_ids`: train 구간 ID 리스트
* `test_ids`: test 구간 ID 리스트

---

### Step 2) Feature Extraction (Image → Numeric Features)

각 샘플에 대해 입력 이미지 3장을 읽고 rail mask 영역만을 사용해 후보 객체를 찾고, **KLT optical flow**로 “정지성”을 정량화합니다.

#### 2.1 Candidate box extraction (rail ROI)

* rail mask로 ROI 제한
* edge/Canny + connected components로 후보 영역(bbox) 생성

#### 2.2 Two-frame matching

* 프레임1 후보 bbox와 프레임2 후보 bbox를 IoU로 매칭 (중복 매칭 방지)

#### 2.3 KLT motion estimation (Lucas-Kanade)

* 매칭된 bbox(또는 union box) 내부에서 Shi-Tomasi 코너를 추출
* LK optical flow로 프레임1→프레임2 이동 벡터를 구해 **median motion** 계산

#### 2.4 Additional descriptors

* bbox 면적 / rail 면적 비율 (`area_ratio`)
* bbox 내부 edge density (`edge_density`)
* stationary score (예: `IoU / (motion + eps)` + area 가중)

#### 2.5 Extracted Feature Vector (per sample)

최종적으로 각 샘플에서 아래 **10개 수치 feature**를 생성합니다.

* `id`
* `num_candidates`
* `min_motion`
* `mean_motion`
* `max_iou`
* `max_area_ratio`
* `stationary_score_1`
* `stationary_score_2`
* `stationary_score_3`
* `edge_density_mean`

산출물:

* `X_train_features`: train ID들에 대한 feature dataframe
* `X_test_features`: test ID들에 대한 feature dataframe

---

### Step 3) Build Training Table (Merge)

CSV는 `ID`, 이미지 feature는 `id`를 사용하므로 컬럼명을 통일한 뒤 ID 기준으로 병합합니다.

* `X_train_features (id → ID)` + `train.csv`의 target 15개 → `train_merged`
* `X_test_features (id → ID)` + `test.csv`의 `ID` → `test_merged`

> 문자열 컬럼(`image-3sec`, `image`, `segmentedRailImage`)은 학습에 필수적이지 않으므로 제외해도 정상입니다.

---

### Step 4) Model Training & Inference

#### 4.1 Why not end-to-end detector?

* 학습 데이터가 약 2천 장 수준이라 대형 detector(YOLO/FasterRCNN)는 과적합 위험
* rail mask가 제공되고, 시간적 정지 특징이 중요 → 고전 CV 기반 feature가 효율적
* 모델 설명 가능성(Feature importance/해석성) 확보

#### 4.2 Two-stage learning (recommended)

1. **Existence classifier**

   * 정지 차량 존재 여부: `max(probaObstacle1..3) > 0`
2. **Multi-output regressor**

   * 존재하는 샘플에 대해 15개 target 회귀

모델:

* `HistGradientBoostingClassifier`
* `HistGradientBoostingRegressor` + `MultiOutputRegressor`

#### 4.3 Test inference & default filling

예측 시 존재하지 않는 샘플은 규칙에 맞게 default 값으로 채움:

* `proba = 0`
* `x = 0.5`
* `y = 0.5`
* `dx = 0`
* `dy = 0`

#### 4.4 Submission formatting

제공된 `sample_submission.csv`의 컬럼 순서를 그대로 사용하여 `submission.csv`를 저장합니다.

산출물:

* `submission.csv`

---

## 4. Outputs

* `X_train_features` / `X_test_features` : image-derived feature tables
* `train_merged` : training table (features + targets)
* `submission.csv` : test prediction results in submission format

---

## 5. Notes (Important)

* ID가 연속이 아닐 수 있으므로 반드시 **폴더 스캔 기반**으로 ID를 구성합니다.
* rail mask 정보는 허용된 feature로 사용합니다.
* “두 이미지가 같은지 여부로 alarm 판단” 같은 규칙 기반 접근은 치팅으로 간주될 수 있으므로 사용하지 않습니다.

