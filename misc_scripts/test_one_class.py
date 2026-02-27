import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier

X = np.random.randn(100, 5)
y = np.zeros(100)  # ONLY ONE CLASS

xgb = XGBClassifier()
xgb.fit(X, y)

explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X)

print("SHAP sum abs:", np.abs(shap_vals).sum())
