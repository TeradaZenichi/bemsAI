import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# Ajuste para imports do projeto
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

# ---- Font configuration for all matplotlib/seaborn plots ----
font_path = 'Gulliver.otf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['font.serif'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Custom font '{prop.get_name()}' loaded for all elements.")
else:
    print("Font 'Gulliver.otf' not found, using Times New Roman.")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['font.serif'] = 'Times New Roman'
    prop = font_manager.FontProperties(family='Times New Roman')

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2, rc={'axes.titlesize': 'large'})

def load_all_results_multi(root_dirs, json_name='costs_rewards_log.json'):
    all_rows = []
    for base_dir in root_dirs:
        if not os.path.isdir(base_dir):
            continue
        for method in sorted(os.listdir(base_dir)):
            subdir = os.path.join(base_dir, method)
            json_path = os.path.join(subdir, json_name)
            if not os.path.isdir(subdir) or not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                log = json.load(f)
            for step, entry in enumerate(log):
                method_name = f"{os.path.basename(base_dir)}/{method}"
                row = {'method': method_name, 'step': step}
                for k, v in entry.items():
                    if isinstance(v, list):
                        row[f"{k}_mean"] = float(sum(v)) / max(len(v), 1)
                        row[f"{k}_last"] = v[-1] if v else None
                    else:
                        row[k] = v
                all_rows.append(row)
    df = pd.DataFrame(all_rows)
    return df

def metric_type(metric):
    metric = metric.lower()
    if 'reward' in metric:
        return 'reward'
    if 'cost' in metric:
        return 'cost'
    return 'cost'

def find_best_per_window(df, metric_field):
    best_rows = []
    typ = metric_type(metric_field)
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if not candidates.empty:
            if typ == 'cost':
                idx = candidates[metric_field].idxmin()
            else:
                idx = candidates[metric_field].idxmax()
            best = candidates.loc[idx]
            best_rows.append({
                'step': step,
                'method': best['method'],
                'best_value': best[metric_field]
            })
    return pd.DataFrame(best_rows)

def podium_ranking(df, metric_field):
    typ = metric_type(metric_field)
    points = {}
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if candidates.empty:
            continue
        if typ == 'cost':
            sorted_group = candidates.sort_values(metric_field, ascending=True)
        else:
            sorted_group = candidates.sort_values(metric_field, ascending=False)
        podium = sorted_group['method'].values[:3]
        for i, pts in enumerate([3,2,1]):
            if i < len(podium):
                method = podium[i]
                points[method] = points.get(method, 0) + pts
    return points

if __name__ == "__main__":
    SCRIPT_DIR = "Paper/"
    method_dirs = [
        os.path.join(SCRIPT_DIR, "PPO_MLP"),
        os.path.join(SCRIPT_DIR, "PPO_MHA"),
        # Add more folders if desired
    ]
    comparison_dir = os.path.join(SCRIPT_DIR, "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    df = load_all_results_multi(method_dirs)
    # ----------- Filter common steps -----------
    method_steps = [set(df[df['method'] == m]['step']) for m in df['method'].unique()]
    common_steps = sorted(set.intersection(*method_steps))
    print(f"Common steps across all methods: {common_steps}")
    df_common = df[df['step'].isin(common_steps)].copy()
    csv_path = os.path.join(comparison_dir, "all_results.csv")
    df_common.to_csv(csv_path, index=False)
    print(f"Summary table (only common windows) saved at: {csv_path}")

    metric_list = [
        'sequential_costs_mean',
        'sequential_rewards_mean',
        'standard_total_cost',
        'standard_total_reward'
    ]

    report_lines = []
    report_lines.append(f"Summary table (only common windows) saved at: {csv_path}\n")

    summary = {}
    all_methods = sorted(df_common['method'].unique())
    all_metrics = metric_list

    # --- Rankings and CSVs ---
    full_score_df = pd.DataFrame(0, index=all_methods, columns=all_metrics)
    podium_score_df = pd.DataFrame(0, index=all_methods, columns=all_metrics)
    podium_summary = {}

    for metric in metric_list:
        if metric not in df_common.columns:
            report_lines.append(f"Column {metric} not found, skipping.")
            continue

        best_df = find_best_per_window(df_common, metric)
        score = best_df['method'].value_counts().sort_values(ascending=False)
        summary[metric] = score
        out_csv = os.path.join(comparison_dir, f'best_{metric}_per_window.csv')
        best_df.to_csv(out_csv, index=False)
        report_lines.append(f"\nBest models per window for {metric} (saved at {out_csv}):")
        report_lines.append(best_df.head().to_string(index=False))

        detailed = "\n".join(
            f"Window {row.step}: {row.method} (value={row.best_value:.4f})"
            for row in best_df.itertuples()
        )
        report_lines.append("\nBest by window:")
        report_lines.append(detailed)

        report_lines.append("\nScore (higher is better):")
        report_lines.append(score.to_string())
        if not score.empty:
            max_score = score.max()
            winners = score[score == max_score].index.tolist()
            winner_msg = f"Winner(s): {', '.join(winners)} [score={max_score}]"
            report_lines.append(winner_msg)

        for method, v in score.items():
            full_score_df.loc[method, metric] = v

        points = podium_ranking(df_common, metric)
        podium_summary[metric] = points
        for method, pts in points.items():
            podium_score_df.loc[method, metric] = pts

    score_full_path = os.path.join(comparison_dir, 'score_full_table.csv')
    full_score_df.to_csv(score_full_path)
    report_lines.append(f"\nClassic score table saved at {score_full_path}")

    podium_path = os.path.join(comparison_dir, 'podium_score_full_table.csv')
    podium_score_df.to_csv(podium_path)
    report_lines.append(f"\nPodium score table saved at {podium_path}")

    report_lines.append("\nREPORT INTERPRETATION")
    report_lines.append("- For each metric, winners are based on whether it is a reward (higher better) or cost (lower better).")
    report_lines.append("- The classic ranking shows how many times each method was 1st place.")
    report_lines.append("- The podium ranking gives 3/2/1 points for 1st/2nd/3rd each window.\n")
    report_lines.append("CLASSIC RANKING TABLE (#windows won per method and metric):\n")
    report_lines.append(full_score_df.to_string())
    report_lines.append("\nPODIUM SCORE TABLE (#points per method and metric):\n")
    report_lines.append(podium_score_df.to_string())
    for metric in metric_list:
        if metric not in podium_score_df.columns:
            continue
        podium_col = podium_score_df[metric]
        max_pts = podium_col.max()
        winners = podium_col[podium_col == max_pts].index.tolist()
        winner_msg = f"Winner(s) (podium) for {metric}: {', '.join(winners)} [score={max_pts}]"
        report_lines.append(winner_msg)

    # --- Stability analysis ---
    report_lines.append("\nSTABILITY: METHOD WITH FEWEST ADVERSE VARIATIONS OVER TIME")
    stability_summary = {}
    for metric in metric_list:
        if metric not in df_common.columns:
            continue
        typ = metric_type(metric)
        adverse_steps = {}
        for method in df_common['method'].unique():
            series = df_common[df_common['method'] == method][metric].reset_index(drop=True)
            if len(series) < 2:
                adverse_steps[method] = None
                continue
            diff = series.diff().fillna(0)
            if typ == 'cost':
                adverse_count = (diff > 0).sum()
            else:
                adverse_count = (diff < 0).sum()
            adverse_steps[method] = adverse_count
        valid_adv = {m: v for m, v in adverse_steps.items() if v is not None}
        if not valid_adv:
            continue
        min_adv = min(valid_adv.values())
        best_methods = [m for m, v in valid_adv.items() if v == min_adv]
        report_lines.append(f"{metric}: {', '.join(best_methods)} ({min_adv} adverse variations)")
        stability_summary[metric] = dict(valid_adv)

    txt_path = os.path.join(comparison_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"\nReport saved at: {txt_path}")

    # --- Top-3/stability line plots ---
    for metric in metric_list:
        if metric not in df_common.columns:
            continue

        typ = metric_type(metric)
        podium_points = podium_score_df[metric]
        top3 = podium_points.sort_values(ascending=False).index[:3].tolist()
        ranks = {m: i+1 for i, m in enumerate(top3)}
        best_per_window = find_best_per_window(df_common, metric)

        # Stability podium calculation for highlight
        adverse_steps = {}
        for method in df_common['method'].unique():
            series = df_common[df_common['method'] == method][metric].reset_index(drop=True)
            if len(series) < 2:
                adverse_steps[method] = None
                continue
            diff = series.diff().fillna(0)
            if typ == 'cost':
                adverse_count = (diff > 0).sum()
            else:
                adverse_count = (diff < 0).sum()
            adverse_steps[method] = adverse_count
        valid_adv = {m: v for m, v in adverse_steps.items() if v is not None}
        sorted_methods = sorted(valid_adv.items(), key=lambda x: x[1])
        stability_points = {m: 0 for m in valid_adv}
        for idx, (m, _) in enumerate(sorted_methods[:3]):
            stability_points[m] = 3 - idx
        max_stab = max(stability_points.values()) if stability_points else 0
        most_stable_methods = [m for m, pts in stability_points.items() if pts == max_stab]

        plt.figure(figsize=(14,7))
        colors = sns.color_palette("muted", 3)
        handles = []
        # Plot top3 podium first
        for i, method in enumerate(top3):
            dat = df_common[df_common['method'] == method]
            x = dat['step']
            y = dat[metric]
            rank = ranks[method]
            label = f"{method} ({rank}{'st' if rank==1 else 'nd' if rank==2 else 'rd'})"
            line, = plt.plot(x, y, label=label, linewidth=2.3, color=colors[(rank-1)%3])
            handles.append(line)
        # Plot most stable (if not already in top3)
        for method in most_stable_methods:
            if method in top3:
                continue
            dat = df_common[df_common['method'] == method]
            x = dat['step']
            y = dat[metric]
            label = f"{method} (Most Stable)"
            line, = plt.plot(x, y, label=label, linewidth=2.5, color='blue', linestyle="--", alpha=0.9)
            handles.append(line)
        # Outros métodos (cinza)
        for method in df_common['method'].unique():
            if method in top3 or method in most_stable_methods:
                continue
            dat = df_common[df_common['method'] == method]
            x = dat['step']
            y = dat[metric]
            plt.plot(x, y, color='gray', alpha=0.18, linewidth=1)

        # Dots vermelhos com texto abaixo do dot
        for _, row in best_per_window.iterrows():
            if row['method'] not in top3 and row['method'] not in most_stable_methods:
                if '/' in row['method']:
                    folder, meth = row['method'].split('/', 1)
                else:
                    folder, meth = '', row['method']
                plt.scatter(row['step'], row['best_value'], color='red', marker='o', s=70, zorder=10)
                full_label = f"{folder}\n{meth}" if folder else meth
                plt.text(row['step'], row['best_value']-0.04*max(1, abs(row['best_value'])),
                         full_label,
                         color='red', fontsize=8, va='top', ha='center', fontweight='bold', linespacing=1.3)

        plt.xlabel('Window (step)')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.legend(handles=handles, frameon=True, ncol=3, bbox_to_anchor=(0.5, 1.01), loc='lower center')

        # Forçar fonte nos elementos
        ax = plt.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(prop)
        ax.xaxis.label.set_fontproperties(prop)
        ax.yaxis.label.set_fontproperties(prop)
        if ax.legend_ is not None:
            for text in ax.legend_.get_texts():
                text.set_fontproperties(prop)

        plt.tight_layout(rect=[0,0,1,0.97])
        outpdf = os.path.join(comparison_dir, f'{metric}_top3_stability_lines.pdf')
        plt.savefig(outpdf, bbox_inches='tight')
        plt.close()
        print(f'Top-3/stability podium plot saved at {outpdf}')
