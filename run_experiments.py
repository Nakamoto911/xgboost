import sys
import os
import time
from datetime import datetime
import pandas as pd
from main import fetch_and_prepare_data, walk_forward_backtest, calculate_metrics, OOS_START_DATE, END_DATE, TARGET_TICKER, LAMBDA_GRID, EWMA_HL_GRID, TRANSACTION_COST
from config import StrategyConfig

# All available experiment configurations
EXPERIMENTS = [
    StrategyConfig(name="1. Paper Baseline"),
    StrategyConfig(name="2. Sortino Tuned", tuning_metric="sortino"),
    StrategyConfig(name="3. Conservative Threshold (0.6)", prob_threshold=0.60),
    StrategyConfig(name="4. Continuous Allocation", allocation_style="continuous"),
    StrategyConfig(name="5. Lambda Smoothing", lambda_smoothing=True),
    StrategyConfig(name="6. Expanding Window", validation_window_type="expanding"),
    StrategyConfig(name="7. Lambda Ensemble (Top 3)", lambda_ensemble_k=3),
    StrategyConfig(
        name="8. The Ultimate Combo",
        tuning_metric="sortino",
        prob_threshold=0.55,
        allocation_style="continuous",
        lambda_smoothing=True
    ),
]


def list_experiments():
    """Print available experiments."""
    print("Available experiments:")
    for i, cfg in enumerate(EXPERIMENTS):
        print(f"  {i + 1}. {cfg.name}")


def parse_experiment_selection(args):
    """Parse CLI arguments to determine which experiments to run.

    Supports:
      - No args or 'all'  -> run all experiments
      - Single number     -> run that experiment (e.g. '3')
      - Comma-separated   -> run those experiments (e.g. '1,3,5')
      - Range             -> run a range (e.g. '2-5')
      - 'list'            -> list experiments and exit

    Returns list of StrategyConfig or None (if 'list' was requested).
    """
    if not args or args[0].lower() == 'all':
        return EXPERIMENTS[:]

    arg = args[0].strip()

    if arg.lower() == 'list':
        list_experiments()
        return None

    indices = set()
    for part in arg.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            for i in range(int(lo), int(hi) + 1):
                indices.add(i)
        else:
            indices.add(int(part))

    selected = []
    for idx in sorted(indices):
        if 1 <= idx <= len(EXPERIMENTS):
            selected.append(EXPERIMENTS[idx - 1])
        else:
            print(f"Warning: experiment {idx} does not exist (valid: 1-{len(EXPERIMENTS)})")

    if not selected:
        print("No valid experiments selected.")
        list_experiments()
        return None

    return selected


def run_experiments(configs):
    """Run selected experiments, compare against B&H, save results."""
    start_time = time.time()
    df = fetch_and_prepare_data()

    # Compute B&H metrics once (using Paper Baseline backtest to get aligned date range)
    print("Computing Buy & Hold baseline...")
    baseline_df = walk_forward_backtest(df, StrategyConfig(name="_baseline"))
    bh_ret, bh_vol, bh_sharpe, bh_sortino, bh_mdd = calculate_metrics(
        baseline_df['Target_Return'], baseline_df['RF_Rate']
    )

    results = []
    for config in configs:
        print(f"\nRunning: {config.name}...")
        res_df = walk_forward_backtest(df, config)
        ret, vol, sharpe, sortino, mdd = calculate_metrics(res_df['Strat_Return'], res_df['RF_Rate'])

        results.append({
            "Experiment": config.name,
            "Ann Ret (%)": round(ret * 100, 2),
            "B&H Ret (%)": round(bh_ret * 100, 2),
            "Ret Delta": round((ret - bh_ret) * 100, 2),
            "Ann Vol (%)": round(vol * 100, 2),
            "Sharpe": round(sharpe, 3),
            "B&H Sharpe": round(bh_sharpe, 3),
            "Sharpe Delta": round(sharpe - bh_sharpe, 3),
            "Sortino": round(sortino, 3),
            "B&H Sortino": round(bh_sortino, 3),
            "Sortino Delta": round(sortino - bh_sortino, 3),
            "Max DD (%)": round(mdd * 100, 2),
            "B&H MDD (%)": round(bh_mdd * 100, 2),
            "MDD Improve": round((mdd - bh_mdd) * 100, 2),
        })

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(results).set_index("Experiment")

    # Print results to console
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS vs BUY & HOLD")
    print("=" * 100)
    print(results_df.to_string())
    print()

    # Summary: which experiments beat B&H on Sharpe
    wins = results_df[results_df['Sharpe Delta'] > 0]
    total = len(results_df)
    n_wins = len(wins)
    print(f"Beat B&H on Sharpe: {n_wins}/{total} experiments ({100 * n_wins / total:.0f}%)")
    if n_wins > 0:
        best = results_df['Sharpe Delta'].idxmax()
        print(f"Best Sharpe improvement: {best} (+{results_df.loc[best, 'Sharpe Delta']:.3f})")
    print(f"\nRuntime: {elapsed:.1f}s")

    # Save markdown report
    save_report(results_df, elapsed)
    return results_df


def save_report(results_df, elapsed):
    """Save experiment results as a timestamped markdown report."""
    os.makedirs("benchmarks", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"benchmarks/experiment_report_{timestamp}.md"

    wins = results_df[results_df['Sharpe Delta'] > 0]
    n_wins = len(wins)
    total = len(results_df)

    lines = []
    lines.append("# Strategy Experiment Report")
    lines.append(f"\n**Generated:** {timestamp}")
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append(f"**Target:** {TARGET_TICKER}")
    lines.append(f"**OOS Period:** {OOS_START_DATE} to {END_DATE}")
    lines.append(f"**Transaction Cost:** {TRANSACTION_COST * 10000:.1f} bps")
    lines.append(f"**Lambda Grid:** {len(LAMBDA_GRID)} candidates")
    lines.append(f"**EWMA Grid:** {EWMA_HL_GRID}")

    # Results table
    lines.append("\n## Results vs Buy & Hold")
    lines.append(f"\n**Beat B&H on Sharpe: {n_wins}/{total} ({100 * n_wins / total:.0f}%)**\n")

    # Header
    cols = ["Sharpe", "B&H Sharpe", "Sharpe Delta", "Ann Ret (%)", "B&H Ret (%)",
            "Sortino", "B&H Sortino", "Max DD (%)", "B&H MDD (%)"]
    header = "| Experiment | " + " | ".join(cols) + " | Verdict |"
    sep = "|---|" + "|".join(["---:" for _ in cols]) + "|---|"
    lines.append(header)
    lines.append(sep)

    for exp, row in results_df.iterrows():
        verdict = "**WIN**" if row['Sharpe Delta'] > 0 else "LOSE"
        delta_str = f"+{row['Sharpe Delta']:.3f}" if row['Sharpe Delta'] > 0 else f"{row['Sharpe Delta']:.3f}"
        vals = [
            f"{row['Sharpe']:.3f}",
            f"{row['B&H Sharpe']:.3f}",
            delta_str,
            f"{row['Ann Ret (%)']:.2f}%",
            f"{row['B&H Ret (%)']:.2f}%",
            f"{row['Sortino']:.3f}",
            f"{row['B&H Sortino']:.3f}",
            f"{row['Max DD (%)']:.2f}%",
            f"{row['B&H MDD (%)']:.2f}%",
        ]
        lines.append(f"| {exp} | " + " | ".join(vals) + f" | {verdict} |")

    # Config details
    lines.append("\n## Experiment Configurations\n")
    for exp, row in results_df.iterrows():
        lines.append(f"- **{exp}**: Sharpe={row['Sharpe']:.3f} (delta={row['Sharpe Delta']:+.3f}), "
                      f"Sortino={row['Sortino']:.3f}, MDD={row['Max DD (%)']:.2f}%")

    report = "\n".join(lines) + "\n"

    with open(filepath, 'w') as f:
        f.write(report)

    print(f"\nReport saved: {filepath}")


def print_usage():
    print("Usage: python run_experiments.py [SELECTION]")
    print()
    print("  SELECTION can be:")
    print("    (empty) or 'all'  - Run all experiments")
    print("    list              - List available experiments")
    print("    N                 - Run experiment N (e.g. '3')")
    print("    N,M,...           - Run specific experiments (e.g. '1,3,5')")
    print("    N-M               - Run a range (e.g. '2-5')")
    print()
    print("Examples:")
    print("  python run_experiments.py           # Run all")
    print("  python run_experiments.py 1         # Run Paper Baseline only")
    print("  python run_experiments.py 1,4,8     # Run experiments 1, 4, and 8")
    print("  python run_experiments.py 2-5       # Run experiments 2 through 5")
    print("  python run_experiments.py list      # Show available experiments")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] in ('-h', '--help', 'help'):
        print_usage()
        sys.exit(0)

    configs = parse_experiment_selection(args)
    if configs is not None:
        run_experiments(configs)
