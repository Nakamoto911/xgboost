import re

with open("app.py", "r") as f:
    content = f.read()

# Add import pickle
if "import pickle" not in content:
    content = content.replace("import os\nimport io", "import os\nimport io\nimport pickle")

# Split at line 113 (which is `    progress_bar.empty()`)
parts = content.split('    progress_bar.empty()\n')

part1 = parts[0] + '    progress_bar.empty()\n'
part2 = parts[1]

cache_save_code = """
    cache_data = {
        'jm_xgb_results': jm_xgb_results,
        'simple_jm_results': simple_jm_results,
        'lambda_history': lambda_history,
        'lambda_dates': lambda_dates,
        'run_simple_jm': run_simple_jm,
        'oos_start_date': backend.OOS_START_DATE,
        'end_date': backend.END_DATE,
    }
    with open('backtest_cache.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
"""

cache_load_code = """
if os.path.exists('backtest_cache.pkl'):
    with open('backtest_cache.pkl', 'rb') as f:
        cache_data = pickle.load(f)
        
    jm_xgb_results = cache_data.get('jm_xgb_results', [])
    simple_jm_results = cache_data.get('simple_jm_results', [])
    lambda_history = cache_data.get('lambda_history', [])
    lambda_dates = cache_data.get('lambda_dates', [])
    run_simple_jm_cached = cache_data.get('run_simple_jm', False)
    cached_oos_start = cache_data.get('oos_start_date', backend.OOS_START_DATE)
    cached_end_date = cache_data.get('end_date', backend.END_DATE)
    
"""

dedented_part2 = "\n".join([line[4:] if line.startswith("    ") else line for line in part2.split("\n")])

# Update variables in dedented_part2
dedented_part2 = dedented_part2.replace('if run_simple_jm:', 'if run_simple_jm_cached:')
dedented_part2 = dedented_part2.replace('backend.TRANSACTION_COST', 'transaction_cost')
dedented_part2 = dedented_part2.replace('backend.OOS_START_DATE', 'cached_oos_start')
dedented_part2 = dedented_part2.replace('backend.END_DATE', 'cached_end_date')

final_content = part1 + cache_save_code + cache_load_code + dedented_part2

with open("app.py", "w") as f:
    f.write(final_content)
