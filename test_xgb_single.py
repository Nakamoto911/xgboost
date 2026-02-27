import numpy as np
import pandas as pd
from xgboost import XGBClassifier

X = np.random.randn(10, 2)
y = np.zeros(10)

xgb = XGBClassifier()
xgb.fit(X, y)
try:
    probs = xgb.predict_proba(X)
    print("Prob shape:", probs.shape)
    print("[:, 1] works?", probs[:, 1])
except Exception as e:
    print("Error:", e)

