"""
Comprehensive Model Evaluation Script for AI-Based Deadline-Aware OS Scheduler
"""
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Adjust plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def plot_xgboost_metrics(stress_results):
    """Plot XGBoost performance metrics under different stress scenarios."""
    scenarios = list(stress_results.keys())
    
    # We will plot regressor MAE and RMSE
    mae_vals = []
    rmse_vals = []
    
    for s in scenarios:
        res = stress_results[s].get('regressor', {})
        mae_vals.append(res.get('MAE', 0))
        rmse_vals.append(res.get('RMSE', 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax.bar(x - width/2, mae_vals, width, label='MAE', color='skyblue')
    ax.bar(x + width/2, rmse_vals, width, label='RMSE', color='salmon')
    
    ax.set_ylabel('Error Value')
    ax.set_title('XGBoost Regressor Error across Stress Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plot_path = PROJECT_ROOT / "results" / "stress_test_results" / "xgb_regressor_errors.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved XGBoost plot: {plot_path}")


def plot_rl_comparison(comparison_metrics):
    """Plot RL vs Heuristics."""
    rows = comparison_metrics.get('comparison_table', [])
    if not rows:
        return
        
    schedulers = [r['scheduler'] for r in rows]
    miss_ratio = [r['deadline_miss_ratio'] for r in rows]
    wait_time = [r['mean_waiting_time'] for r in rows]
    turnaround = [r['mean_turnaround_time'] for r in rows]
    cpu_util = [r['cpu_utilization'] for r in rows]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    sns.barplot(x=schedulers, y=miss_ratio, ax=axs[0], palette="Blues_d")
    axs[0].set_title('Deadline Miss Ratio (Lower is Better)')
    axs[0].set_ylabel('Ratio')
    
    sns.barplot(x=schedulers, y=wait_time, ax=axs[1], palette="Greens_d")
    axs[1].set_title('Mean Waiting Time (Lower is Better)')
    axs[1].set_ylabel('Time units')
    
    sns.barplot(x=schedulers, y=turnaround, ax=axs[2], palette="Oranges_d")
    axs[2].set_title('Mean Turnaround Time (Lower is Better)')
    axs[2].set_ylabel('Time units')

    sns.barplot(x=schedulers, y=cpu_util, ax=axs[3], palette="Purples_d")
    axs[3].set_title('CPU Utilization (Higher is usually Better)')
    axs[3].set_ylabel('Utilization Ratio')
    
    plt.suptitle("RL Policy vs Heuristics Performance on Test Set", fontsize=16)
    plt.tight_layout()
    
    plot_path = PROJECT_ROOT / "results" / "stress_test_results" / "rl_vs_heuristics.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved RL comparison plot: {plot_path}")


def main():
    xgb_stress_file = PROJECT_ROOT / "results" / "stress_test_results" / "stress_test_results.json"
    rl_eval_file = PROJECT_ROOT / "artifacts" / "rl" / "stress_beat_sjf_v1_20260410_retry" / "evaluation" / "comparison_metrics.json"
    
    if xgb_stress_file.exists():
        with open(xgb_stress_file) as f:
            data = json.load(f)
            if 'xgboost' in data:
                plot_xgboost_metrics(data['xgboost'])
    else:
        print("XGBoost stress test results not found. Generating...")
        from stress_test_model import stress_test_xgboost, load_xgboost_models
        clf, reg = load_xgboost_models()
        res = stress_test_xgboost(clf, reg)
        plot_xgboost_metrics(res)
                
    if rl_eval_file.exists():
        with open(rl_eval_file) as f:
            data = json.load(f)
            plot_rl_comparison(data)
    else:
        print("RL test results not found!")

if __name__ == "__main__":
    main()
