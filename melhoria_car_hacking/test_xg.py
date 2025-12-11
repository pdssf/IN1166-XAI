
import shap
import xgboost as xgboost
from intrusion_detection_ch import load_data, XGBoost_train
from utils_chd import load_malicious_data, load_benign_data

data_chd = load_data()
X_malicious, y_malicious = load_malicious_data()
model = XGBoost_train(data_chd)

explainer = shap.TreeExplainer(model, feature_perturbation = "interventional", model_output='raw')
malicious_shap_values = explainer(shap.sample(X_malicious, 10))
benign_shap_values = explainer(shap.sample(load_benign_data()[0], 10))

print("Benign SHAP Values Waterfall Plots:")
for v in benign_shap_values:
    shap.plots.waterfall(v)

print("Malicious SHAP Values Waterfall Plots:")
for v in malicious_shap_values:
    shap.plots.waterfall(v)
