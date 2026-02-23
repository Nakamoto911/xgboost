import pandas as pd
import numpy as np

try:
    with open('data_cache.pkl', 'rb') as f:
        pass
except:
    pass

import os
# The data is in results_cache.pkl? Let's check where the backtest results are.
# In app.py, what is the filename? It's "run_simple_jm_cached".
import app
