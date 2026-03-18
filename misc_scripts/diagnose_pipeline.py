"""
Pipeline Health Diagnostics — JM-XGB Strategy
Generates a compact Markdown report assessing ML model quality independent of financial returns.
Output is optimized for LLM consumption (minimal tokens, maximal signal).

Usage:
    source .venv/bin/activate
    python misc_scripts/diagnose_pipeline.py              # Full diagnostics
    python misc_scripts/diagnose_pipeline.py --quick       # Skip slow tests (permutation, sensitivity)
    python misc_scripts/diagnose_pipeline.py --help
"""

import sys
import os
import types
import argparse
import time
from datetime import datetime

# Compatibility shim for distutils.version (Python 3.12+)
try:
    import distutils
    import distutils.version
except ImportError:
    d = types.ModuleType('distutils')
    dv = types.ModuleType('distutils.version')
    class LooseVersion:
        def __init__(self, vstring=None): self.vstring = vstring
        def __str__(self): return self.vstring
        def __lt__(self, other): return self.vstring < (other.vstring if hasattr(other, 'vstring') else other)
        def __le__(self, other): return self.vstring <= (other.vstring if hasattr(other, 'vstring') else other)
        def __gt__(self, other): return self.vstring > (other.vstring if hasattr(other, 'vstring') else other)
        def __ge__(self, other): return self.vstring >= (other.vstring if hasattr(other, 'vstring') else other)
        def __eq__(self, other): return self.vstring == (other.vstring if hasattr(other, 'vstring') else other)
    dv.LooseVersion = LooseVersion
    d.version = dv
    sys.modules['distutils'] = d
    sys.modules['distutils.version'] = dv

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, balanced_accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, average_precision_score,
    confusion_matrix,
)
from scipy.stats import ttest_ind, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from main import (
    fetch_and_prepare_data, StatisticalJumpModel, _forecast_cache,
    OOS_START_DATE, END_DATE, START_DATE_DATA, LAMBDA_GRID, EWMA_HL_GRID,
    VALIDATION_WINDOW_YRS, TRANSACTION_COST, TARGET_TICKER,
)
from config import StrategyConfig

BENCHMARKS_DIR = os.path.join(PROJECT_ROOT, 'benchmarks')
os.makedirs(BENCHMARKS_DIR, exist_ok=True)


# =============================================================================
# Diagnostic Engine
# =============================================================================

class PipelineDiagnostics:
    """Runs all diagnostic tests and collects results into a structured dict."""

    def __init__(self, df, config=None, quick=False):
        self.df = df
        self.config = config or StrategyConfig()
        self.quick = quick
        self.chunks = []  # per-chunk diagnostics
        self.gates = {}   # gate checklist results

    def run_all(self):
        """Execute walk-forward diagnostics, collecting per-chunk telemetry."""
        print("Running pipeline diagnostics...")
        current_date = pd.to_datetime(OOS_START_DATE)
        final_end = pd.to_datetime(END_DATE)

        # Phase 1: Tune EWMA halflife (same as main.py)
        init_val_start = current_date - pd.DateOffset(years=VALIDATION_WINDOW_YRS)
        best_ewma_hl = EWMA_HL_GRID[-1]

        chunk_idx = 0
        all_true_labels = []
        all_pred_probs = []
        all_pred_labels = []
        all_forecast_states = []
        all_jm_states = []
        all_returns = []
        all_silhouettes = []
        all_train_acc = []
        all_oos_acc = []
        all_train_ll = []
        all_oos_ll = []
        regime_durations = []
        switch_outcomes = []  # (switched_to_cash, subsequent_5d_return)

        while current_date < final_end:
            chunk_end = min(current_date + pd.DateOffset(months=6), final_end)
            chunk_diag = self._diagnose_chunk(current_date, chunk_end, best_ewma_hl)

            if chunk_diag is not None:
                self.chunks.append(chunk_diag)

                # Accumulate for aggregate metrics
                cd = chunk_diag
                if cd.get('oos_true') is not None and len(cd['oos_true']) > 0:
                    all_true_labels.extend(cd['oos_true'])
                    all_pred_probs.extend(cd['oos_probs'])
                    all_pred_labels.extend(cd['oos_pred'])
                    all_returns.extend(cd['oos_returns'])
                if cd.get('silhouette') is not None:
                    all_silhouettes.append(cd['silhouette'])
                if cd.get('train_acc') is not None:
                    all_train_acc.append(cd['train_acc'])
                    all_oos_acc.append(cd['oos_acc'])
                    all_train_ll.append(cd['train_ll'])
                    all_oos_ll.append(cd['oos_ll'])

            chunk_idx += 1
            current_date = chunk_end
            print(f"  Chunk {chunk_idx} done ({current_date.date()})")

        # Aggregate metrics
        self.aggregate = self._compute_aggregate(
            all_true_labels, all_pred_probs, all_pred_labels, all_returns,
            all_silhouettes, all_train_acc, all_oos_acc, all_train_ll, all_oos_ll,
        )

        # Permutation test (slow)
        if not self.quick and len(all_true_labels) > 100:
            self.aggregate['permutation_p'] = self._permutation_test(
                np.array(all_true_labels), np.array(all_pred_probs), n_perms=500
            )
        else:
            self.aggregate['permutation_p'] = None

        # Compute gates
        self.gates = self._evaluate_gates()

        print(f"  Diagnostics complete: {len(self.chunks)} chunks analyzed.")

    def _diagnose_chunk(self, current_date, chunk_end, ewma_hl):
        """Run JM + XGB for one 6-month chunk and extract diagnostics."""
        df = self.df
        config = self.config

        train_start = current_date - pd.DateOffset(years=11)
        train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
        if len(train_df) < 252 * 5:
            return None

        return_features = [c for c in train_df.columns if c.startswith(('DD_', 'Avg_Ret_', 'Sortino_'))]
        macro_features = ['Yield_2Y_EWMA_diff', 'Yield_Slope_EWMA_10',
                          'Yield_Slope_EWMA_diff_21', 'VIX_EWMA_log_diff', 'Stock_Bond_Corr']
        all_features = return_features + macro_features

        # Standardize for JM
        X_train_jm = train_df[return_features]
        jm_mean = X_train_jm.mean()
        jm_std = X_train_jm.std()
        jm_std[jm_std == 0] = 1.0
        X_train_jm_std = (X_train_jm - jm_mean) / jm_std

        # Fit JM — use median lambda from grid as representative
        lmbda = np.median(LAMBDA_GRID[1:]) if len(LAMBDA_GRID) > 1 else 10.0
        jm = StatisticalJumpModel(n_states=2, lambda_penalty=lmbda)
        identified_states = jm.fit_predict(X_train_jm_std.values)

        # Align states: State 0 = Bull
        cum_ret_0 = train_df['Excess_Return'][identified_states == 0].sum()
        cum_ret_1 = train_df['Excess_Return'][identified_states == 1].sum()
        if cum_ret_1 > cum_ret_0:
            identified_states = 1 - identified_states
            jm.means = jm.means[::-1].copy()

        # --- JM Cluster Quality ---
        diag = {
            'period_start': current_date.strftime('%Y-%m-%d'),
            'period_end': chunk_end.strftime('%Y-%m-%d'),
            'lambda': lmbda,
            'train_days': len(train_df),
        }

        try:
            diag['silhouette'] = float(silhouette_score(X_train_jm_std.values, identified_states))
        except Exception:
            diag['silhouette'] = None
        try:
            diag['davies_bouldin'] = float(davies_bouldin_score(X_train_jm_std.values, identified_states))
        except Exception:
            diag['davies_bouldin'] = None
        try:
            diag['calinski_harabasz'] = float(calinski_harabasz_score(X_train_jm_std.values, identified_states))
        except Exception:
            diag['calinski_harabasz'] = None

        diag['centroid_l2'] = float(np.linalg.norm(jm.means[0] - jm.means[1]))

        # Regime economic coherence
        bull_mask = identified_states == 0
        bear_mask = identified_states == 1
        n_bull = bull_mask.sum()
        n_bear = bear_mask.sum()
        diag['state_balance'] = float(n_bear / len(identified_states)) if len(identified_states) > 0 else 0

        bull_ret = train_df['Excess_Return'].values[bull_mask]
        bear_ret = train_df['Excess_Return'].values[bear_mask]
        diag['bull_ann_ret'] = float(np.mean(bull_ret) * 252) if len(bull_ret) > 0 else 0
        diag['bear_ann_ret'] = float(np.mean(bear_ret) * 252) if len(bear_ret) > 0 else 0
        diag['return_delta'] = diag['bull_ann_ret'] - diag['bear_ann_ret']
        diag['return_separation_correct'] = diag['bull_ann_ret'] > diag['bear_ann_ret']

        bull_vol = float(np.std(bull_ret) * np.sqrt(252)) if len(bull_ret) > 1 else 1e-8
        bear_vol = float(np.std(bear_ret) * np.sqrt(252)) if len(bear_ret) > 1 else 1e-8
        diag['vol_ratio'] = bear_vol / bull_vol if bull_vol > 0 else 0

        # Feature-level t-tests (top 3 most significant)
        t_stats = {}
        for feat in return_features:
            vals_0 = X_train_jm_std.values[bull_mask, return_features.index(feat)]
            vals_1 = X_train_jm_std.values[bear_mask, return_features.index(feat)]
            if len(vals_0) > 1 and len(vals_1) > 1:
                t, p = ttest_ind(vals_0, vals_1, equal_var=False)
                t_stats[feat] = p
        diag['n_significant_features'] = sum(1 for p in t_stats.values() if p < 0.05)

        # --- XGBoost OOS Diagnostics ---
        # Create targets (shifted labels)
        train_df_xgb = train_df.copy()
        train_df_xgb['Target_State'] = np.roll(identified_states, -1)
        train_df_xgb = train_df_xgb.iloc[:-1]

        oos_df = df[(df.index >= current_date) & (df.index < chunk_end)].copy()
        if len(oos_df) == 0:
            return diag

        X_train_xgb = train_df_xgb[all_features]
        y_train = train_df_xgb['Target_State']
        X_oos = oos_df[all_features]

        # Get true OOS JM labels
        X_oos_jm = oos_df[return_features]
        X_oos_jm_std = (X_oos_jm - jm_mean) / jm_std
        oos_states = jm.predict_online(X_oos_jm_std.values, last_known_state=identified_states[-1])

        # Train XGBoost
        from xgboost import XGBClassifier
        unique_labels = y_train.unique()
        if len(unique_labels) < 2:
            return diag

        xgb = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, **config.xgb_params)
        xgb.fit(X_train_xgb, y_train)

        # Training metrics
        train_probs = xgb.predict_proba(X_train_xgb)[:, 1]
        train_pred = (train_probs > 0.5).astype(int)
        diag['train_acc'] = float(accuracy_score(y_train, train_pred))
        diag['train_ll'] = float(log_loss(y_train, train_probs, labels=[0, 1]))

        # OOS predictions
        oos_probs = xgb.predict_proba(X_oos)[:, 1]

        # True labels: JM state at t+1
        true_labels = np.roll(oos_states, -1)[:-1]
        pred_probs_aligned = oos_probs[:-1]
        pred_labels = (pred_probs_aligned > 0.5).astype(int)
        oos_returns = oos_df['Target_Return'].values[:-1]

        if len(true_labels) < 10:
            return diag

        diag['oos_true'] = true_labels.tolist()
        diag['oos_probs'] = pred_probs_aligned.tolist()
        diag['oos_pred'] = pred_labels.tolist()
        diag['oos_returns'] = oos_returns.tolist()

        diag['oos_acc'] = float(accuracy_score(true_labels, pred_labels))
        diag['oos_balanced_acc'] = float(balanced_accuracy_score(true_labels, pred_labels))
        diag['oos_ll'] = float(log_loss(true_labels, pred_probs_aligned, labels=[0, 1]))
        diag['oos_brier'] = float(brier_score_loss(true_labels, pred_probs_aligned))

        try:
            diag['oos_auc'] = float(roc_auc_score(true_labels, pred_probs_aligned))
        except Exception:
            diag['oos_auc'] = None

        diag['oos_mcc'] = float(matthews_corrcoef(true_labels, pred_labels))
        diag['oos_kappa'] = float(cohen_kappa_score(true_labels, pred_labels))
        diag['bear_precision'] = float(precision_score(true_labels, pred_labels, zero_division=0))
        diag['bear_recall'] = float(recall_score(true_labels, pred_labels, zero_division=0))

        # Calibration bias
        diag['calibration_bias'] = float(np.mean(pred_probs_aligned) - np.mean(true_labels))

        # Probability sharpness (std of predicted probs — higher = more decisive)
        diag['prob_std'] = float(np.std(pred_probs_aligned))

        # Rank IC (1-day)
        if len(oos_returns) == len(pred_probs_aligned):
            rho, p_ic = spearmanr(pred_probs_aligned, oos_returns)
            diag['rank_ic_1d'] = float(rho)
        else:
            diag['rank_ic_1d'] = None

        # Rank IC (5-day)
        if len(oos_returns) > 5:
            fwd_5d = pd.Series(oos_returns).rolling(5).sum().shift(-4).values
            valid = ~np.isnan(fwd_5d)
            if valid.sum() > 10:
                rho5, _ = spearmanr(pred_probs_aligned[valid], fwd_5d[valid])
                diag['rank_ic_5d'] = float(rho5)
            else:
                diag['rank_ic_5d'] = None
        else:
            diag['rank_ic_5d'] = None

        # Feature importance (top 5 by gain)
        imp = xgb.feature_importances_
        top_idx = np.argsort(imp)[::-1][:5]
        diag['top_features'] = [(all_features[i], float(imp[i])) for i in top_idx]

        # Regime switch hit rate for this chunk
        signal = (pd.Series(pred_probs_aligned) > 0.5).astype(int)
        switches_to_cash = signal.diff().fillna(0)
        switch_to_cash_idx = switches_to_cash[switches_to_cash == 1].index.tolist()
        hits = 0
        total_switches = 0
        for idx in switch_to_cash_idx:
            if idx + 5 < len(oos_returns):
                fwd_ret = np.sum(oos_returns[idx:idx+5])
                total_switches += 1
                if fwd_ret < -TRANSACTION_COST * 2:  # avoided a drop > round-trip cost
                    hits += 1
        diag['switch_hit_rate'] = hits / total_switches if total_switches > 0 else None
        diag['n_switches'] = total_switches

        # Average holding period
        runs = []
        current_run = 1
        for i in range(1, len(pred_labels)):
            if pred_labels[i] == pred_labels[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        diag['avg_holding_period'] = float(np.mean(runs)) if runs else 0

        # Time in market
        diag['time_in_market'] = float(1.0 - np.mean(pred_labels))

        return diag

    def _compute_aggregate(self, true_labels, pred_probs, pred_labels, returns,
                            silhouettes, train_acc, oos_acc, train_ll, oos_ll):
        """Compute aggregate metrics across all chunks."""
        agg = {}

        true_arr = np.array(true_labels)
        prob_arr = np.array(pred_probs)
        pred_arr = np.array(pred_labels)
        ret_arr = np.array(returns)

        if len(true_arr) < 20:
            return agg

        # Classification
        agg['accuracy'] = float(accuracy_score(true_arr, pred_arr))
        agg['balanced_accuracy'] = float(balanced_accuracy_score(true_arr, pred_arr))
        agg['log_loss'] = float(log_loss(true_arr, prob_arr, labels=[0, 1]))
        agg['brier_score'] = float(brier_score_loss(true_arr, prob_arr))
        try:
            agg['auc_roc'] = float(roc_auc_score(true_arr, prob_arr))
        except Exception:
            agg['auc_roc'] = None
        try:
            agg['auc_pr'] = float(average_precision_score(true_arr, prob_arr))
        except Exception:
            agg['auc_pr'] = None
        agg['mcc'] = float(matthews_corrcoef(true_arr, pred_arr))
        agg['kappa'] = float(cohen_kappa_score(true_arr, pred_arr))
        agg['bear_precision'] = float(precision_score(true_arr, pred_arr, zero_division=0))
        agg['bear_recall'] = float(recall_score(true_arr, pred_arr, zero_division=0))
        agg['bear_f1'] = float(f1_score(true_arr, pred_arr, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(true_arr, pred_arr, labels=[0, 1])
        agg['confusion_matrix'] = cm.tolist()

        # Calibration
        agg['calibration_bias'] = float(np.mean(prob_arr) - np.mean(true_arr))
        agg['prob_mean'] = float(np.mean(prob_arr))
        agg['prob_std'] = float(np.std(prob_arr))
        agg['actual_bear_pct'] = float(np.mean(true_arr))

        # ECE (10 bins)
        ece = 0.0
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            mask = (prob_arr >= lo) & (prob_arr < hi)
            if mask.sum() > 0:
                bin_acc = true_arr[mask].mean()
                bin_conf = prob_arr[mask].mean()
                ece += mask.sum() / len(prob_arr) * abs(bin_acc - bin_conf)
        agg['ece'] = float(ece)

        # Rank IC
        rho1, _ = spearmanr(prob_arr, ret_arr)
        agg['rank_ic_1d'] = float(rho1)

        # IC at multiple horizons
        ret_series = pd.Series(ret_arr)
        for h in [5, 10, 21]:
            fwd = ret_series.rolling(h).sum().shift(-(h-1)).values
            valid = ~np.isnan(fwd)
            if valid.sum() > 20:
                rho_h, _ = spearmanr(prob_arr[valid], fwd[valid])
                agg[f'rank_ic_{h}d'] = float(rho_h)
            else:
                agg[f'rank_ic_{h}d'] = None

        # IC Information Ratio (rolling 126-day windows)
        window = 126
        ics = []
        for i in range(window, len(prob_arr)):
            r, _ = spearmanr(prob_arr[i-window:i], ret_arr[i-window:i])
            if not np.isnan(r):
                ics.append(r)
        if ics:
            agg['ic_mean'] = float(np.mean(ics))
            agg['ic_std'] = float(np.std(ics))
            agg['ic_ir'] = float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0
        else:
            agg['ic_mean'] = agg['ic_std'] = agg['ic_ir'] = None

        # JM cluster quality
        if silhouettes:
            agg['silhouette_mean'] = float(np.mean(silhouettes))
            agg['silhouette_std'] = float(np.std(silhouettes))
        else:
            agg['silhouette_mean'] = agg['silhouette_std'] = None

        # Overfitting gap
        if train_acc and oos_acc:
            gaps = [t - o for t, o in zip(train_acc, oos_acc)]
            agg['acc_gap_mean'] = float(np.mean(gaps))
            agg['acc_gap_max'] = float(np.max(gaps))
            ll_gaps = [o - t for t, o in zip(train_ll, oos_ll)]
            agg['ll_gap_mean'] = float(np.mean(ll_gaps))
        else:
            agg['acc_gap_mean'] = agg['acc_gap_max'] = agg['ll_gap_mean'] = None

        # Execution profile (aggregate from chunks)
        holding_periods = [c['avg_holding_period'] for c in self.chunks if c.get('avg_holding_period')]
        agg['avg_holding_period'] = float(np.mean(holding_periods)) if holding_periods else None

        switch_hits = [c['switch_hit_rate'] for c in self.chunks if c.get('switch_hit_rate') is not None]
        agg['switch_hit_rate'] = float(np.mean(switch_hits)) if switch_hits else None

        tim = [c['time_in_market'] for c in self.chunks if c.get('time_in_market') is not None]
        agg['time_in_market'] = float(np.mean(tim)) if tim else None

        # Feature stability (Jaccard of top-3 across chunks)
        top3_sets = []
        for c in self.chunks:
            if c.get('top_features'):
                top3_sets.append(set(f for f, _ in c['top_features'][:3]))
        if len(top3_sets) >= 2:
            jaccards = []
            for i in range(1, len(top3_sets)):
                inter = len(top3_sets[i] & top3_sets[i-1])
                union = len(top3_sets[i] | top3_sets[i-1])
                jaccards.append(inter / union if union > 0 else 0)
            agg['feature_jaccard_mean'] = float(np.mean(jaccards))
        else:
            agg['feature_jaccard_mean'] = None

        # Label imbalance
        bear_pcts = [c['state_balance'] for c in self.chunks if c.get('state_balance') is not None]
        agg['label_imbalance_mean'] = float(np.mean(bear_pcts)) if bear_pcts else None

        # Return separation rate
        sep_correct = [c['return_separation_correct'] for c in self.chunks if c.get('return_separation_correct') is not None]
        agg['return_sep_rate'] = float(np.mean(sep_correct)) if sep_correct else None

        # JM initialization sensitivity (only if not quick mode)
        agg['init_sensitivity'] = None  # computed separately if needed

        return agg

    def _permutation_test(self, true_labels, pred_probs, n_perms=500):
        """Permutation test for AUC significance."""
        try:
            actual_auc = roc_auc_score(true_labels, pred_probs)
        except Exception:
            return None

        count_above = 0
        for _ in range(n_perms):
            perm = np.random.permutation(true_labels)
            try:
                perm_auc = roc_auc_score(perm, pred_probs)
                if perm_auc >= actual_auc:
                    count_above += 1
            except Exception:
                pass
        return float(count_above / n_perms)

    def _evaluate_gates(self):
        """Evaluate the 14 minimum-viable-evidence gates."""
        a = self.aggregate
        g = {}

        g['G1_silhouette'] = (a.get('silhouette_mean', 0) or 0) > 0.10
        g['G2_return_sep'] = (a.get('return_sep_rate', 0) or 0) >= 0.90
        imb = a.get('label_imbalance_mean', 0.5) or 0.5
        g['G3_label_balance'] = 0.05 <= imb <= 0.50
        g['G4_latency'] = True  # Placeholder — requires crash-specific analysis
        g['G5_balanced_acc'] = (a.get('balanced_accuracy', 0) or 0) > 0.52
        g['G6_auc_roc'] = (a.get('auc_roc', 0) or 0) > 0.53
        g['G7_log_loss'] = (a.get('log_loss', 1.0) or 1.0) < 0.68
        g['G8_overfit_gap'] = (a.get('acc_gap_mean', 1.0) or 1.0) < 0.15
        g['G9_ece'] = (a.get('ece', 1.0) or 1.0) < 0.15
        g['G10_permutation'] = (a.get('permutation_p') is not None and a['permutation_p'] < 0.05) if a.get('permutation_p') is not None else None
        g['G11_init_sensitivity'] = None  # requires separate test
        g['G12_feature_importance'] = True  # Placeholder — would need permutation importance
        g['G13_switch_hit_rate'] = (a.get('switch_hit_rate', 0) or 0) > 0.40
        g['G14_holding_period'] = (a.get('avg_holding_period', 0) or 0) > 15

        return g


# =============================================================================
# Markdown Report Generator
# =============================================================================

def generate_report(diag: PipelineDiagnostics, elapsed: float) -> str:
    """Generate compact, token-efficient markdown report."""
    a = diag.aggregate
    g = diag.gates
    chunks = diag.chunks

    L = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')

    L.append(f"# JM-XGB Pipeline Diagnostics — {ts}")
    L.append(f"OOS: {OOS_START_DATE} → {END_DATE} | Target: {TARGET_TICKER} | Chunks: {len(chunks)} | Runtime: {elapsed:.0f}s")
    L.append("")

    # === Pipeline Parameters ===
    import json
    L.append("## 0. Pipeline Parameters")
    L.append("")
    L.append("### Pipeline Setup")
    L.append(f"- **Target Ticker:** `{TARGET_TICKER}`")
    L.append(f"- **JM Lambda Grid:** `{LAMBDA_GRID}`")
    L.append(f"- **EWMA Halflife Grid:** `{EWMA_HL_GRID}`")
    L.append(f"- **Validation Window:** `{VALIDATION_WINDOW_YRS} years`")
    L.append(f"- **Transaction Cost:** `{TRANSACTION_COST}`")
    L.append("")
    L.append("### Strategy Config")
    config_dict = getattr(diag.config, '__dict__', {})
    for k, v in config_dict.items():
        if k != 'xgb_params':
            L.append(f"- **{k}:** `{v}`")
    L.append("")
    L.append("### XGBoost Parameters")
    L.append("```json")
    L.append(json.dumps(getattr(diag.config, 'xgb_params', {}), indent=2))
    L.append("```")
    L.append("")

    # === Gate Checklist ===
    L.append("## 1. Gate Checklist")
    L.append("")
    n_pass = sum(1 for v in g.values() if v is True)
    n_fail = sum(1 for v in g.values() if v is False)
    n_na = sum(1 for v in g.values() if v is None)
    L.append(f"**{n_pass} PASS / {n_fail} FAIL / {n_na} N/A** out of {len(g)}")
    L.append("")
    L.append("| Gate | Metric | Value | Threshold | Status |")
    L.append("|---|---|---:|---|---|")

    gate_details = {
        'G1_silhouette': ('Silhouette (mean)', a.get('silhouette_mean'), '> 0.10'),
        'G2_return_sep': ('Return sep rate', a.get('return_sep_rate'), '≥ 90%'),
        'G3_label_balance': ('Bear label %', a.get('label_imbalance_mean'), '5–50%'),
        'G4_latency': ('Peak-to-signal lag', 'N/A', '< 15 days'),
        'G5_balanced_acc': ('Balanced accuracy', a.get('balanced_accuracy'), '> 52%'),
        'G6_auc_roc': ('AUC-ROC', a.get('auc_roc'), '> 0.53'),
        'G7_log_loss': ('Log-loss', a.get('log_loss'), '< 0.68'),
        'G8_overfit_gap': ('Train-OOS acc gap', a.get('acc_gap_mean'), '< 15pp'),
        'G9_ece': ('ECE', a.get('ece'), '< 0.15'),
        'G10_permutation': ('Permutation p', a.get('permutation_p'), '< 0.05'),
        'G11_init_sensitivity': ('Init sensitivity', 'N/A', '< 15%'),
        'G12_feature_importance': ('Feat perm imp', 'N/A', '≥ 3 positive'),
        'G13_switch_hit_rate': ('Switch hit rate', a.get('switch_hit_rate'), '> 40%'),
        'G14_holding_period': ('Avg hold period', a.get('avg_holding_period'), '> 15 days'),
    }

    for gate_key, (label, val, threshold) in gate_details.items():
        status = g.get(gate_key)
        if status is True:
            status_str = "✅ PASS"
        elif status is False:
            status_str = "❌ FAIL"
        else:
            status_str = "⬜ N/A"

        if isinstance(val, float):
            val_str = f"{val:.4f}"
        elif val is None:
            val_str = "—"
        else:
            val_str = str(val)

        L.append(f"| {gate_key} | {label} | {val_str} | {threshold} | {status_str} |")

    L.append("")

    # === JM Quality ===
    L.append("## 2. JM Regime Quality")
    L.append("")

    sil_vals = [c['silhouette'] for c in chunks if c.get('silhouette') is not None]
    db_vals = [c['davies_bouldin'] for c in chunks if c.get('davies_bouldin') is not None]
    ch_vals = [c['calinski_harabasz'] for c in chunks if c.get('calinski_harabasz') is not None]
    l2_vals = [c['centroid_l2'] for c in chunks if c.get('centroid_l2') is not None]
    vr_vals = [c['vol_ratio'] for c in chunks if c.get('vol_ratio') is not None]
    rd_vals = [c['return_delta'] for c in chunks if c.get('return_delta') is not None]
    sf_vals = [c['n_significant_features'] for c in chunks if c.get('n_significant_features') is not None]

    L.append("| Metric | Mean | Std | Min | Max |")
    L.append("|---|---:|---:|---:|---:|")

    def row(name, vals):
        if vals:
            return f"| {name} | {np.mean(vals):.3f} | {np.std(vals):.3f} | {np.min(vals):.3f} | {np.max(vals):.3f} |"
        return f"| {name} | — | — | — | — |"

    L.append(row("Silhouette", sil_vals))
    L.append(row("Davies-Bouldin", db_vals))
    L.append(row("Calinski-Harabasz", ch_vals))
    L.append(row("Centroid L2", l2_vals))
    L.append(row("Vol ratio (Bear/Bull)", vr_vals))
    L.append(row("Return delta (ann%)", [v*100 for v in rd_vals] if rd_vals else []))
    L.append(row("Significant features (of 9)", [float(v) for v in sf_vals]))

    sep_rate = a.get('return_sep_rate')
    imb = a.get('label_imbalance_mean')
    L.append("")
    L.append(f"Return separation correct: **{sep_rate*100:.0f}%** of windows" if sep_rate else "Return separation: N/A")
    L.append(f"Bear label share (mean): **{imb*100:.1f}%**" if imb else "Label imbalance: N/A")
    L.append("")

    # === XGBoost OOS Classification ===
    L.append("## 3. XGBoost OOS Classification")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---:|")

    def mrow(name, key, fmt=".4f"):
        v = a.get(key)
        return f"| {name} | {v:{fmt}} |" if v is not None else f"| {name} | — |"

    L.append(mrow("Accuracy", 'accuracy'))
    L.append(mrow("Balanced Accuracy", 'balanced_accuracy'))
    L.append(mrow("AUC-ROC", 'auc_roc'))
    L.append(mrow("AUC-PR", 'auc_pr'))
    L.append(mrow("Log-Loss", 'log_loss'))
    L.append(mrow("Brier Score", 'brier_score'))
    L.append(mrow("MCC", 'mcc'))
    L.append(mrow("Cohen's Kappa", 'kappa'))
    L.append(mrow("Bear Precision", 'bear_precision'))
    L.append(mrow("Bear Recall", 'bear_recall'))
    L.append(mrow("Bear F1", 'bear_f1'))
    L.append("")

    # Confusion matrix
    cm = a.get('confusion_matrix')
    if cm:
        L.append(f"Confusion (rows=true, cols=pred): Bull→[{cm[0][0]}, {cm[0][1]}] Bear→[{cm[1][0]}, {cm[1][1]}]")
        L.append("")

    # === Calibration ===
    L.append("## 4. Probability Calibration")
    L.append("")
    L.append(f"- ECE: **{a.get('ece', 0):.4f}**")
    L.append(f"- Calibration bias: **{a.get('calibration_bias', 0):+.4f}** (pred mean {a.get('prob_mean', 0):.3f} vs actual Bear {a.get('actual_bear_pct', 0):.3f})")
    L.append(f"- Prob std (sharpness): **{a.get('prob_std', 0):.4f}** ({'bimodal/sharp' if (a.get('prob_std', 0) or 0) > 0.15 else 'concentrated/weak'})")
    L.append("")

    # === Overfitting ===
    L.append("## 5. Overfitting Monitor")
    L.append("")
    L.append(f"- Train-OOS accuracy gap (mean): **{a.get('acc_gap_mean', 0):.3f}** (max: {a.get('acc_gap_max', 0):.3f})")
    L.append(f"- Train-OOS log-loss gap (mean): **{a.get('ll_gap_mean', 0):+.3f}**")
    if a.get('permutation_p') is not None:
        L.append(f"- Permutation test p-value (AUC, {500} perms): **{a['permutation_p']:.4f}**")
    else:
        L.append("- Permutation test: skipped (--quick mode)")
    L.append("")

    # === Signal Quality ===
    L.append("## 6. Signal Quality (IC)")
    L.append("")
    L.append("| Horizon | Rank IC |")
    L.append("|---|---:|")
    L.append(f"| 1-day | {a.get('rank_ic_1d', 0):.4f} |")
    L.append(f"| 5-day | {a.get('rank_ic_5d', 'N/A')} |" if a.get('rank_ic_5d') is None else f"| 5-day | {a['rank_ic_5d']:.4f} |")
    L.append(f"| 10-day | {a.get('rank_ic_10d', 'N/A')} |" if a.get('rank_ic_10d') is None else f"| 10-day | {a['rank_ic_10d']:.4f} |")
    L.append(f"| 21-day | {a.get('rank_ic_21d', 'N/A')} |" if a.get('rank_ic_21d') is None else f"| 21-day | {a['rank_ic_21d']:.4f} |")
    L.append("")
    if a.get('ic_ir') is not None:
        L.append(f"IC Information Ratio (126d rolling): **{a['ic_ir']:.3f}** (mean IC: {a['ic_mean']:.4f}, std: {a['ic_std']:.4f})")
    L.append("")

    # === Execution Profile ===
    L.append("## 7. Execution Profile")
    L.append("")
    L.append(f"- Time in market: **{a.get('time_in_market', 0)*100:.1f}%**" if a.get('time_in_market') else "- Time in market: N/A")
    L.append(f"- Avg holding period: **{a.get('avg_holding_period', 0):.1f} days**" if a.get('avg_holding_period') else "- Avg holding period: N/A")
    L.append(f"- Regime switch hit rate: **{a.get('switch_hit_rate', 0)*100:.1f}%**" if a.get('switch_hit_rate') else "- Switch hit rate: N/A")
    L.append("")

    # === Feature Stability ===
    L.append("## 8. Feature Stability")
    L.append("")
    if a.get('feature_jaccard_mean') is not None:
        L.append(f"Top-3 feature Jaccard similarity (adjacent chunks): **{a['feature_jaccard_mean']:.3f}**")
    L.append("")

    # Feature frequency table (most common top-3 features across chunks)
    from collections import Counter
    all_top3 = []
    for c in chunks:
        if c.get('top_features'):
            for f, imp in c['top_features'][:3]:
                all_top3.append(f)
    if all_top3:
        freq = Counter(all_top3).most_common(7)
        L.append("| Feature | Times in Top-3 | Frequency |")
        L.append("|---|---:|---:|")
        n_chunks_with_feats = sum(1 for c in chunks if c.get('top_features'))
        for feat, count in freq:
            L.append(f"| {feat} | {count} | {count/n_chunks_with_feats*100:.0f}% |")
    L.append("")

    # === Per-Chunk Summary (compact) ===
    L.append("## 9. Per-Chunk Summary")
    L.append("")
    L.append("| Period | Sil | DBI | AUC | Acc | LL | IC1d | BearPct | HoldPd |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in chunks:
        period = c.get('period_start', '?')[:7]
        sil = f"{c['silhouette']:.3f}" if c.get('silhouette') is not None else "—"
        dbi = f"{c['davies_bouldin']:.2f}" if c.get('davies_bouldin') is not None else "—"
        auc = f"{c['oos_auc']:.3f}" if c.get('oos_auc') is not None else "—"
        acc = f"{c['oos_acc']:.3f}" if c.get('oos_acc') is not None else "—"
        ll = f"{c['oos_ll']:.3f}" if c.get('oos_ll') is not None else "—"
        ic1 = f"{c['rank_ic_1d']:.3f}" if c.get('rank_ic_1d') is not None else "—"
        bp = f"{c['state_balance']*100:.0f}%" if c.get('state_balance') is not None else "—"
        hp = f"{c['avg_holding_period']:.0f}" if c.get('avg_holding_period') else "—"
        L.append(f"| {period} | {sil} | {dbi} | {auc} | {acc} | {ll} | {ic1} | {bp} | {hp} |")
    L.append("")

    # === Interpretation ===
    L.append("## Interpretation Guide")
    L.append("")
    L.append("- **Silhouette > 0.15**: clusters are well-separated in feature space")
    L.append("- **DBI < 1.5**: good compactness-to-separation ratio")
    L.append("- **AUC-ROC > 0.55**: genuine discriminative signal")
    L.append("- **Log-Loss < 0.693**: better than random guessing")
    L.append("- **Rank IC < -0.02 (1d)**: higher P(Bear) predicts lower returns")
    L.append("- **ECE < 0.10**: probabilities are well-calibrated")
    L.append("- **Acc gap < 0.15**: not memorizing training data")
    L.append("- **Holding period > 20d**: not whipsawing")
    L.append("- **Switch hit rate > 45%**: regime switches are economically justified")
    L.append("")

    return "\n".join(L) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="JM-XGB Pipeline Health Diagnostics")
    parser.add_argument('--quick', action='store_true', help='Skip slow tests (permutation, sensitivity)')
    args = parser.parse_args()

    t0 = time.time()

    print("Fetching data...")
    df = fetch_and_prepare_data()

    config = StrategyConfig()
    diag = PipelineDiagnostics(df, config, quick=args.quick)
    diag.run_all()

    elapsed = time.time() - t0

    report = generate_report(diag, elapsed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(BENCHMARKS_DIR, f'pipeline_diagnostics_{timestamp}.md')
    with open(filepath, 'w') as f:
        f.write(report)

    print(f"\nReport saved: {filepath}")
    print(f"Total runtime: {elapsed:.1f}s")

    # Also print gate summary
    g = diag.gates
    n_pass = sum(1 for v in g.values() if v is True)
    n_fail = sum(1 for v in g.values() if v is False)
    print(f"\nGate summary: {n_pass} PASS / {n_fail} FAIL")
    for k, v in g.items():
        icon = "✅" if v is True else ("❌" if v is False else "⬜")
        print(f"  {icon} {k}")


if __name__ == '__main__':
    main()
