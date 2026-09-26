# 📊 Regression Analysis & Insights Report: California Housing Valuation

**Task**: Regression Model Prediction  
**Dataset**: California Housing Dataset (20,640 census block groups, 8 continuous predictors, 1 continuous target)  
**Models Evaluated**: 
1. Baseline Simple Linear Regression (`MedInc`)
2. Multiple Linear Regression (OLS with all 8 standardized features)
3. Polynomial Regression (Degree 2 with $L_2$ Ridge Regularization, $\alpha = 50.0$)

---

## 1. Executive Summary

This study formulates an end-to-end regression modeling framework to predict median house values (`MedHouseVal`) across California block groups based on demographic, geographic, and structural housing metrics. 

### Core Results:
- **Baseline Simple Linear Regression** achieves $R^2 = 0.4740$, validating that district income level is the single primary driver of real estate prices.
- **Multiple Linear Regression (OLS)** incorporates all 8 features, increasing $R^2$ to **0.5758** ($\text{RMSE} = \$74,558$, $\text{MAE} = \$53,320$).
- **Polynomial Regression (Degree 2 + Ridge Regularization)** captures non-linear geographic interactions ($\text{Latitude} \times \text{Longitude}$) and wealth concentration, elevating $R^2$ to **0.6673** ($\text{RMSE} = \$66,028$, $\text{MAE} = \$47,080$).
- This represents an improvement of **+9.15 percentage points of explained variance** and reduces mean prediction error by **\$6,240 per home** over linear regression.

---

## 2. Quantitative Model Performance Summary

All models were evaluated using an 80/20 train/test split ($N_{\text{train}} = 16,512$, $N_{\text{test}} = 4,128$) with a fixed random seed (`random_state=42`). Metrics are expressed in native units (hundreds of thousands of dollars, where $1.0 = \$100,000$):

| Model Architecture | Train $R^2$ | Test $R^2$ | Train MAE | Test MAE | Train RMSE | Test RMSE | Generalization Gap ($\Delta R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Linear Regression (`MedInc`)** | 0.4735 | 0.4740 | 0.6279 (\$62.8k) | 0.6264 (\$62.6k) | 0.8385 (\$83.9k) | 0.8306 (\$83.1k) | -0.0005 |
| **Multiple Linear Regression (All Features)** | 0.6126 | 0.5758 | 0.5286 (\$52.9k) | 0.5332 (\$53.3k) | 0.7197 (\$72.0k) | 0.7456 (\$74.6k) | +0.0368 |
| **Polynomial Regression (Deg 2 Ridge, $\alpha=50$)** | **0.6816** | **0.6673** | **0.4675** (\$46.8k) | **0.4708** (\$47.1k) | **0.6524** (\$65.2k) | **0.6603** (\$66.0k) | **+0.0143** |

---

## 3. Visual Analysis of Predictions vs. Actual Values

### 3.1 Prediction Alignment ($y$ vs. $\hat{y}$)
- **Multiple Linear Regression**: Scatter points align broadly around the identity line ($y = x$), but exhibit significant vertical dispersion across the mid-range (\$150,000 to \$350,000). Linear regression also predicts impossible negative prices ($\hat{y} < 0$) for a small number of economically depressed districts.
- **Polynomial Regression**: Substantially tightens the scatter envelope along the identity diagonal. The variance is noticeably reduced in the \$200,000 to \$400,000 bracket, demonstrating superior local curvature fitting.

### 3.2 Residual Diagnostics & Homoscedasticity
- **Fitted vs. Residual Plot**: In both models, the residuals are broadly centered along the horizontal zero error line ($\mu_{\text{res}} \approx -0.002$). However, two clear non-ideal patterns emerge:
  1. **The \$500,000 Ceiling Truncation Line**: A distinct downward-sloping linear pattern of negative residuals appears for actual values equal to 5.0. Because the U.S. Census Bureau capped median home values at \$500,000, multi-million dollar homes were recorded as 5.0. The models legitimately estimate higher values (\$5.5k–\$8.0k) based on income and coastal location, producing an artificial residual artifact ($y - \hat{y} < 0$).
  2. **Fan-shaped Heteroscedasticity**: Error variance expands moderately as home values increase. Lower-priced homes exhibit tight clustering, whereas high-end homes feature unmeasured amenities (view, luxury finishes, prestige school districts) that linear/polynomial models cannot observe.
- **Residual Distribution**: The error distribution closely follows a Gaussian bell curve, verifying that the fundamental Gauss-Markov assumptions of regression are largely satisfied.

### 3.3 Trajectory Tracking
- Plotting actual values against predictions over a sequential slice of 75 test homes shows that Polynomial Regression tracks sharp local peaks and valleys significantly faster than linear regression, avoiding the sluggish smoothing of linear hyperplanes.

---

## 4. Key Real Estate & Economic Insights

### 4.1 Dominant Drivers of Valuation (Standardized $\beta$ Weights)
1. **Median Income (`MedInc`, $\beta = +0.854$, $r = +0.690$)**:
   - By a substantial margin, neighborhood purchasing power is the single strongest predictor. Holding all other features constant, an increase of 1 standard deviation in district median income (~\\$19,000) corresponds to a **+\$98,500** increase in median home value.
2. **Geographic Gradient (`Latitude` $\beta = -0.887$, `Longitude` $\beta = -0.858$)**:
   - Individually, increasing latitude (moving northward) and longitude (moving eastward/inland) depresses home prices. Coastal Southern California and the Bay Area represent high-value enclaves compared to the agricultural Central Valley.
3. **Neighborhood Maturity (`HouseAge`, $\beta = +0.123$, $r = +0.106$)**:
   - Older homes command a premium, reflecting established neighborhoods with mature foliage, historical architectural value, and prime central municipal locations.
4. **Room vs. Bedroom Multi-collinearity (`AveRooms` $\beta = -0.666$, `AveBedrms` $\beta = +0.648$)**:
   - `AveRooms` and `AveBedrms` are strongly correlated ($r = 0.848$). The opposing signs indicate structural density: when holding total room count constant, having more dedicated bedrooms indicates greater usable living space and tenant capacity.

---

## 5. Non-Linearity, Bias-Variance Tradeoff, and Regularization

- **Why Degree 2 Polynomial Works**: Expanding 8 features to degree 2 generates 44 features. The cross-terms (such as $\text{Latitude} \times \text{Longitude}$ and $\text{MedInc} \times \text{AveRooms}$) enable the model to capture geographic hot-spots that cannot be expressed by separate linear terms.
- **Why Regularization Was Critical**: Unregularized OLS on 44 polynomial features causes numerical ill-conditioning due to multicollinear interaction terms. Applying **Ridge Regularization ($\alpha = 50.0$)** shrinks non-essential interaction weights, reducing test RMSE from \$0.6814 (unregularized) to **\$0.6603**, while shrinking the train-test generalization gap to just **0.0143**.

---

## 6. Recommendations & Next Steps

1. **Censored Regression (Tobit Model)**: To eliminate the ceiling artifact at \$500k, a Tobit model or survival/censored regression formulation should be adopted to formally model right-censored observations.
2. **Geospatial Feature Engineering**: Rather than pure polynomial coordinates, creating explicit spatial distances (e.g., *Haversine distance to coast*, *distance to San Francisco/Silicon Valley*, *distance to Los Angeles*) would capture geographical economics directly.
3. **Ensemble Tree Models**: Gradient Boosted Trees (XGBoost / LightGBM) would excel on this dataset by learning non-linear, non-smooth decision boundaries for discrete school districts and municipal lines without feature expansion.
