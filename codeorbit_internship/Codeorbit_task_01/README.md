# Executive Report: Titanic Dataset Preprocessing & Exploratory Data Analysis

**Author:** Antigravity Data Science & Analytics  
**Dataset:** Titanic: Machine Learning from Disaster (Kaggle / UCI Repository Benchmark)  
**Deliverables:** Cleaned Dataset (`data/processed/titanic_cleaned.csv`), Execution Pipeline (`run_pipeline.py`), Fully-Executed Jupyter Notebook (`data_preprocessing_and_eda.ipynb`), High-Resolution Figures (`figures/`)

---

## 1. Executive Summary

This study conducts an end-to-end data preparation, cleaning, and exploratory data analysis (EDA) pipeline on the Titanic passenger survival dataset. The primary objective is to clean the dataset from imperfections (missing values, duplicate risks, extreme outliers), synthesize predictive features, and uncover sociological, demographic, and economic drivers governing passenger survival.

### Key Discoveries:
1. **Gender Disparity ("Women and Children First"):** Female passengers exhibited a **74.20% survival rate** versus **18.89% for male passengers** ($p < 0.001$, Pearson $r = +0.54$).
2. **Socio-Economic Stratification:** 1st Class passengers experienced a **62.96% survival rate**, compared to **47.28%** in 2nd Class and **24.24%** in 3rd Class. Wealth and cabin deck level strongly dictated lifeboat access.
3. **The "Sweet Spot" of Family Dynamics:** Traveling completely alone had a lower survival rate (**30.35%**). Moderate family sizes (2 to 4 members) achieved the highest survival probability (**55% to 72%**), whereas large families ($\ge 5$) suffered catastrophic survival drops (< 20%).
4. **Data Normalization & Outlier Suppression:** Original ticket fares exhibited extreme positive skewness (**4.79**) and kurtosis (**33.40**), with luxury suites exceeding $500. Applying 99th-percentile Winsorization and logarithmic scaling normalized skewness to **0.34**, eliminating extreme leverage points while retaining economic signal.

---

## 2. Dataset Architecture & Initial Assessment

The raw dataset contains **891 passenger records** across **12 attributes** comprising numerical, categorical, and text identifiers:

| Feature Name | Raw Data Type | Missing Count | Missing % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `PassengerId` | `int64` | 0 | 0.00% | Unique passenger integer identifier |
| `Survived` | `int64` | 0 | 0.00% | Target variable (0 = Deceased, 1 = Survived) |
| `Pclass` | `int64` | 0 | 0.00% | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | `object` | 0 | 0.00% | Full passenger name with formal title |
| `Sex` | `object` | 0 | 0.00% | Biological sex (`male`, `female`) |
| `Age` | `float64` | 177 | 19.87% | Age in years (fractional for infants) |
| `SibSp` | `int64` | 0 | 0.00% | Count of siblings/spouses aboard |
| `Parch` | `int64` | 0 | 0.00% | Count of parents/children aboard |
| `Ticket` | `object` | 0 | 0.00% | Ticket serial number |
| `Fare` | `float64` | 0 | 0.00% | Passenger fare paid (?) |
| `Cabin` | `object` | 687 | 77.10% | Cabin room allocation |
| `Embarked` | `object` | 2 | 0.22% | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 3. Data Cleaning & Preprocessing Methodology

### 3.1 Duplicate Record Auditing
- **Exact Row Duplicates:** Verified $0$ duplicated rows across all columns.
- **Primary Key Integrity:** Verified $0$ duplicated `PassengerId` values, confirming strictly unique entity records.

### 3.2 Handling Missing Values
Three attributes contained null values, each handled with domain-informed methodologies:

1. **`Embarked` (2 missing records, 0.22%):**
   - Imputed using the empirical mode: `'S'` (Southampton), which accounts for 72.50% of all passenger boardings.
2. **`Age` (177 missing records, 19.87%):**
   - A naive global median or mean introduces severe artificial variance suppression.
   - **Methodology:** We extracted formal passenger titles from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, and `Rare`). We then calculated and imputed the grouped median age conditioned on `(Title, Pclass)`. For example, `Master` (young boys) had a median age of ~4 years, while `Mr` in 1st Class had a median age of 40 years. This granular imputation preserved biological reality and life-stage variance.
3. **`Cabin` (687 missing records, 77.10%):**
   - Categorized as **Missing Not At Random (MNAR)**: 3rd Class passengers were largely unassigned specific cabin identifiers.
   - Rather than dropping the column or using arbitrary imputation, we engineered two high-value features:
     - `Has_Cabin`: Binary indicator ($1$ if cabin recorded, $0$ if null).
     - `Deck`: Extracted initial cabin character (`A`, `B`, `C`, `D`, `E`, `F`, `G`, `T`, or `'Unknown'`).

### 3.3 Outlier Detection & Treatment
Using Tukey's Interquartile Range (IQR) method:
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

- **`Fare`:**
  - $Q_1 = 7.91$, $Q_3 = 31.00$, $\text{IQR} = 23.09$. Upper fence = $65.63$.
  - 116 records (13.02%) exceeded the fence, with extreme outliers reaching $512.33$.
  - **Treatment:** We applied **99th-percentile Winsorization** (capping at $249.01$) to retain relative wealth distinctions without destabilizing gradient calculations. Furthermore, we generated `Fare_Log = log1p(Fare_Capped)` which compressed skewness from **4.79** down to **0.34**.
- **`Age`:**
  - $Q_1 = 21.00$, $Q_3 = 36.75$, $\text{IQR} = 15.75$. Upper fence = $60.38$.
  - 22 passengers were identified above 60 years of age (up to 80 years).
  - **Treatment:** These records reflect authentic historical passengers rather than measurement errors. Because survival varies distinctly among seniors, these values were retained in full.

### 3.4 Feature Engineering
To maximize analytical depth, four new features were derived:
- `FamilySize = SibSp + Parch + 1` (Total travel group size).
- `IsAlone = 1` if `FamilySize == 1`, else `0`.
- `AgeGroup`: Discretized cohorts: *Child (0-12)*, *Adolescent (12-18)*, *Young Adult (18-35)*, *Middle-Aged (35-60)*, *Senior (60+)*.
- `Title`: Parsed honorifics (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`).

---

## 4. Comprehensive Summary Statistics

### 4.1 Numerical Features (Post-Cleaning)

| Feature | Count | Mean | Std | Min | 25% (Q1) | Median (50%) | 75% (Q3) | Max | IQR | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Age`** (Imputed) | 891 | 29.14 | 13.49 | 0.42 | 21.00 | 26.00 | 36.75 | 80.00 | 15.75 | 0.47 | 0.59 |
| **`Age_Original`** | 714 | 29.70 | 14.53 | 0.42 | 20.12 | 28.00 | 38.00 | 80.00 | 17.88 | 0.39 | 0.18 |
| **`Fare`** (Raw) | 891 | 32.20 | 49.69 | 0.00 | 7.91 | 14.45 | 31.00 | 512.33 | 23.09 | 4.79 | 33.40 |
| **`Fare_Capped`** | 891 | 31.22 | 42.52 | 0.00 | 7.91 | 14.45 | 31.00 | 249.01 | 23.09 | 3.11 | 11.03 |
| **`Fare_Log`** | 891 | 2.96 | 0.96 | 0.00 | 2.19 | 2.74 | 3.47 | 5.52 | 1.28 | 0.34 | 0.80 |
| **`SibSp`** | 891 | 0.52 | 1.10 | 0.00 | 0.00 | 0.00 | 1.00 | 8.00 | 1.00 | 3.70 | 17.88 |
| **`Parch`** | 891 | 0.38 | 0.81 | 0.00 | 0.00 | 0.00 | 0.00 | 6.00 | 0.00 | 2.75 | 9.78 |
| **`FamilySize`** | 891 | 1.90 | 1.61 | 1.00 | 1.00 | 1.00 | 2.00 | 11.00 | 1.00 | 2.73 | 9.16 |

### 4.2 Categorical Distributions & Survival Cross-Tabulation

| Feature | Category | Count | Proportion (%) | Survival Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **`Survived`** | 0 (Died) | 549 | 61.62% | 0.00% |
| | 1 (Survived) | 342 | 38.38% | 100.00% |
| **`Pclass`** | 1st Class | 216 | 24.24% | 62.96% |
| | 2nd Class | 184 | 20.65% | 47.28% |
| | 3rd Class | 491 | 55.11% | 24.24% |
| **`Sex`** | Female | 314 | 35.24% | 74.20% |
| | Male | 577 | 64.76% | 18.89% |
| **`Embarked`** | Southampton (S) | 646 | 72.50% | 33.90% |
| | Cherbourg (C) | 168 | 18.86% | 55.36% |
| | Queenstown (Q) | 77 | 8.64% | 38.96% |
| **`Title`** | Mr | 517 | 58.02% | 15.67% |
| | Miss | 185 | 20.76% | 70.27% |
| | Mrs | 126 | 14.14% | 79.37% |
| | Master | 40 | 4.49% | 57.50% |
| | Rare | 23 | 2.58% | 34.78% |
| **`AgeGroup`** | Child (0-12) | 73 | 8.19% | 57.53% |
| | Adolescent (12-18) | 103 | 11.56% | 47.57% |
| | Young Adult (18-35) | 469 | 52.64% | 33.05% |
| | Middle-Aged (35-60) | 224 | 25.14% | 40.62% |
| | Senior (60+) | 22 | 2.47% | 22.73% |
| **`IsAlone`** | Solo (1) | 537 | 60.27% | 30.35% |
| | With Family (0) | 354 | 39.73% | 50.56% |
| **`Has_Cabin`** | No Cabin (0) | 687 | 77.10% | 29.99% |
| | Has Cabin (1) | 204 | 22.90% | 66.67% |

---

## 5. Visual Insights & Exploratory Data Analysis

### 5.1 Missingness Profile & Resolution (`figures/01_missing_values_profile.png`)
The initial missingness audit revealed heavy missingness in `Cabin` (77.1%) and moderate missingness in `Age` (19.9%). Following the multi-stage imputation strategy, all 9 critical analytical columns achieve **100% data completeness (0 nulls)** without introducing synthetic mode bias.

### 5.2 Outlier Dynamics & Distribution Normalization (`figures/02_outlier_detection_treatment.png`)
Boxplot and KDE evaluations reveal the severe right-skewness of raw fares. Winsorization at the 99th percentile ($249.01$) effectively compressed extreme values while preserving the ordinal variance between 1st, 2nd, and 3rd classes. Furthermore, the grouped median imputation of age seamlessly tracks the original kernel density curve without producing artificial spikes.

### 5.3 Univariate Demographic Distributions (`figures/03_univariate_distributions.png`)
- The baseline survival rate was **38.38%** (342 survivors vs 549 casualties).
- The passenger population was predominantly male (**64.76%**) and 3rd Class (**55.11%**).
- 60.27% of passengers were traveling solo (`IsAlone = 1`).

### 5.4 Bivariate Drivers of Survival (`figures/04_bivariate_survival_factors.png`)
- **Gender:** Female survival (74.20%) vs. Male survival (18.89%) was the dominant partition.
- **Class:** 1st Class survival (62.96%) was 2.6 times higher than 3rd Class survival (24.24%).
- **Embarkation:** Cherbourg passengers experienced a markedly higher survival rate (55.36%), attributable to a higher proportion of 1st-class ticket holders boarding in Cherbourg.
- **Age Cohort:** Children (<12 yrs) achieved a 57.53% survival rate, whereas Seniors (60+) had the lowest survival rate at 22.73%.

### 5.5 Correlation Matrix Analysis (`figures/05_correlation_matrix.png`)
- `Sex (Female=1)` correlates highest with `Survived` ($r = +0.54$).
- `Pclass` is strongly negatively correlated with `Survived` ($r = -0.34$) and `Fare_Log` ($r = -0.68$).
- `Fare_Log` has a positive correlation with survival ($r = +0.33$).
- `Has_Cabin` has a notable positive correlation with survival ($r = +0.32$), indicating the physical advantage of upper-deck staterooms.

### 5.6 Multivariate Intersectional Effects (`figures/06_multivariate_interactions.png`)
- **Class $\times$ Gender Interaction:**
  - 1st Class Female: **96.8%** survival rate.
  - 2nd Class Female: **92.1%** survival rate.
  - 3rd Class Female: **50.0%** survival rate.
  - 1st Class Male: **36.9%** survival rate.
  - 2nd Class Male: **15.7%** survival rate.
  - 3rd Class Male: **13.5%** survival rate.
- **Family Size $\times$ Gender:**
  - Solo females had a 78.6% survival rate; females in families of 2 to 4 members exceeded 80% survival.
  - Males in families of 2 to 4 members improved their survival to ~28%, but plunged to 0% in large families of $\ge 8$.

---

## 6. Recommendations for Downstream Machine Learning

1. **Tree-Based & Ensemble Models:** Random Forest and XGBoost will benefit from nonlinear features like `FamilySize`, `Deck`, `Title`, and `AgeGroup`.
2. **Linear & Logistic Models:** Standardize numerical variables using `StandardScaler` on `Age` and `Fare_Log` to ensure gradient stability.
3. **High Cardinality Features:** One-hot encode `Embarked` and `Title`, while discarding raw `Name`, `Ticket`, and `PassengerId`.

---

## 7. Execution & Artifact Verification

All scripts, notebooks, and outputs are fully reproducible:
- **Execute complete pipeline:**
  ```powershell
  python run_pipeline.py
  ```
- **Inspect Jupyter Notebook:**
  Open `data_preprocessing_and_eda.ipynb` in VS Code or Jupyter Lab.
