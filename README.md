Heart Disease Prediction Analysis
This project uses machine learning models, Logistic Regression and XGBoost, to predict the presence of heart disease based on clinical diagnostic data. The analysis includes feature engineering, skewness handling, and model interpretability using SHAP values to uncover non-linear relationships and clinical interactions.This project was done as part of Kaggle competition .
Summary
The XGBoost Classifier outperformed Logistic Regression.
XGBoost Accuracy: 88.7%
XGBoost AUC: 0.95
Logistic Regression Accuracy:88.4%
Logistic Regression AUC:0.94
Key Features of the Analysis
1.Advanced Feature Engineering
1.Risk Indicator:Created a custom interaction feature based on the negative correlation between Thallium and Max HR. high Thallium (>6) combined with lower Max HR (<150) was flagged as a high-risk state.
2.skewness Correction:Applied `log1p` transformations to ST Depression and Cholesterol to normalize highly skewed distributions.
3.Feature Scaling:Standardized numerical features (BP,Max HR,Age) using `StandardScaler` to ensure model stability.

2. Model Interpretability (SHAP)

Used SHAP (SHLinearly Explainer & TreeExplainer)to explain "black-box" predictions and understand impact of features:

1.Identified Thallium, Chest Pain Type, Max HR as the primary features that drives model output.
2.Linear vs. Non-Linear: Observed that Logistic Regression treats categorical variables (Sex, Thallium) linearly, whereas XGBoost captures complex spreads, indicating deep interactions with other patient variables.
3.Discovered a significant interaction between Sex and Thallium.
4.While women generally have lower baseline risk, a "Reversible Defect" (Thallium 7) in females acts as a significantly stronger risk signal than in males, likely indicating more advanced disease progression.

*Language:Python
* Libraries: Pandas, NumPy, Scikit-Learn, XGBoost
* Visualization: Matplotlib, Seaborn, SHAP
Project Structure
* `train.csv` / `test.csv`: Clinical datasets.
* `heart_disease_analysis.ipynb`: The complete pipeline from EDA to model deployment.
* `README.md`: Project overview.
Results & Visualizations
The project includes several  plots:
1.Correlation Heatmaps: Showing relationships between diagnostic tests.
2.Waterfall Plots: Detailed breakdown of individual patient predictions.
3.Dependence Plots: Visualizing how risk increases as specific metrics (like Max HR) change.
---

### How to Run

1. Clone the repository.
2. Ensure you have the required libraries installed:
```bash
pip install pandas scikit-learn xgboost shap matplotlib seaborn

```
3. Run the Jupyter Notebook to view the full analysis and SHAP interpretations.

