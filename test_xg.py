
import shap
import xgboost as xgboost

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

model = xgboost.XGBClassifier(objective="binary:logistic", seed=10).fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model, feature_perturbation = "interventional", model_output='probability')
shap_values = explainer(X)

# visualize the first prediction's explanation

shap.plots.waterfall(shap_values[0])

