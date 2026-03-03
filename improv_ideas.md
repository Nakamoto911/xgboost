## Ideas for Further Improvement

### Within paper methodology
1. **Sortino-based walk-forward objective** — Replace Sharpe with Sortino for lambda selection. The strategy's edge is drawdown reduction; Sortino better captures downside risk avoidance and may produce more stable lambda selections.

2. **Walk-forward lambda smoothing** — Instead of using the single best lambda per period, use an exponentially weighted average of recent best lambdas to prevent sudden jumps. E.g., `lambda_t = 0.7 * best_lambda_t + 0.3 * lambda_{t-1}`.

3. **Expanding validation window** — The paper uses a fixed 5-year rolling window. An expanding window (starting from 5 years, growing over time) would give more data for later periods and may produce more stable selections.

4. **Feature selection per period** — Not all 13 features are informative in all market regimes. Walk-forward feature importance could drop low-value features to reduce XGB noise.

5. **Calibrated probability threshold** — Instead of fixed 0.5, use isotonic regression or Platt scaling on validation probabilities to calibrate the threshold. Our tests showed threshold=0.55-0.60 can improve Sharpe.

### Extensions beyond the paper
6. **Partial allocation** — Instead of binary 0/1 switching, use the smoothed probability as a continuous allocation weight (e.g., invest `1 - P(bear)` fraction in equities). This reduces the cost of false signals.

7. **Multi-asset portfolio** — The paper's full framework uses 12 assets with mean-variance optimization. A single asset is the hardest case; adding bonds/REITs/commodities would enable cross-asset diversification of regime signals.

8. **Ensemble of lambdas** — Instead of picking one best lambda, average the signals from top-K lambdas on the validation window to produce a more robust ensemble forecast.

9. **Regime-dependent risk parity** — Instead of 100% equity or 100% risk-free, allocate between equity/bonds/cash based on regime confidence using risk parity weights.

10. **Online learning for XGBoost** — Instead of refitting from scratch every 6 months, use incremental training to adapt more quickly to regime changes while retaining memory of older patterns.
