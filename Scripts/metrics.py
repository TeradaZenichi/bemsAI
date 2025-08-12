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

# ----------------------------- Helpers -----------------------------
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
    m = metric.lower()
    if 'reward' in m:
        return 'reward'  # maior é melhor
    if 'cost' in m:
        return 'cost'    # menor é melhor
    return 'cost'

def find_best_per_window(df, metric_field):
    best_rows = []
    typ = metric_type(metric_field)
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if candidates.empty:
            continue
        idx = candidates[metric_field].idxmin() if typ == 'cost' else candidates[metric_field].idxmax()
        best = candidates.loc[idx]
        best_rows.append({'step': step, 'method': best['method'], 'best_value': best[metric_field]})
    return pd.DataFrame(best_rows)

def podium_ranking(df, metric_field):
    typ = metric_type(metric_field)
    points = {}
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if candidates.empty:
            continue
        sorted_group = candidates.sort_values(metric_field, ascending=(typ == 'cost'))
        podium = sorted_group['method'].values[:3]
        for i, pts in enumerate([3, 2, 1]):
            if i < len(podium):
                m = podium[i]
                points[m] = points.get(m, 0) + pts
    return points

def compute_stability_points(df, metric):
    """Retorna dict method-> pontos de estabilidade (3/2/1 para menor nº de variações adversas)."""
    typ = metric_type(metric)
    adverse_steps = {}
    for method in df['method'].unique():
        series = df[df['method'] == method][metric].reset_index(drop=True)
        if len(series) < 2:
            continue
        diff = series.diff().fillna(0)
        adverse_count = (diff > 0).sum() if typ == 'cost' else (diff < 0).sum()
        adverse_steps[method] = int(adverse_count)
    if not adverse_steps:
        return {}
    sorted_methods = sorted(adverse_steps.items(), key=lambda x: x[1])  # menos adverso primeiro
    points = {m: 0 for m in adverse_steps}
    for idx, (m, _) in enumerate(sorted_methods[:3]):
        points[m] = 3 - idx  # 3,2,1
    return points

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    SCRIPT_DIR = "Paper/"
    method_dirs = [
        os.path.join(SCRIPT_DIR, "PPO_MLP"),
        os.path.join(SCRIPT_DIR, "PPO_MHA"),
        # adicione mais pastas se desejar
    ]
    comparison_dir = os.path.join(SCRIPT_DIR, "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    df = load_all_results_multi(method_dirs)

    # ---- steps comuns a todos os métodos ----
    method_steps = [set(df[df['method'] == m]['step']) for m in df['method'].unique()]
    common_steps = sorted(set.intersection(*method_steps)) if method_steps else []
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

    # -------- Relatório + tabelas de ranking clássico e pódio --------
    report_lines = [f"Summary table (only common windows) saved at: {csv_path}\n"]
    all_methods = sorted(df_common['method'].unique())
    full_score_df = pd.DataFrame(0, index=all_methods, columns=metric_list)
    podium_score_df = pd.DataFrame(0, index=all_methods, columns=metric_list)

    for metric in metric_list:
        if metric not in df_common.columns:
            report_lines.append(f"Column {metric} not found, skipping.")
            continue

        best_df = find_best_per_window(df_common, metric)
        score = best_df['method'].value_counts().sort_values(ascending=False)
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
            report_lines.append(f"Winner(s): {', '.join(winners)} [score={max_score}]")

        for method, v in score.items():
            full_score_df.loc[method, metric] = v

        points = podium_ranking(df_common, metric)
        for method, pts in points.items():
            podium_score_df.loc[method, metric] = pts

    score_full_path = os.path.join(comparison_dir, 'score_full_table.csv')
    full_score_df.to_csv(score_full_path)
    report_lines.append(f"\nClassic score table saved at {score_full_path}")

    podium_path = os.path.join(comparison_dir, 'podium_score_full_table.csv')
    podium_score_df.to_csv(podium_path)
    report_lines.append(f"\nPodium score table saved at {podium_path}")

    # -------- Análise de estabilidade (texto) --------
    report_lines.append("\nSTABILITY: METHOD WITH FEWEST ADVERSE VARIATIONS OVER TIME")
    for metric in metric_list:
        if metric not in df_common.columns:
            continue
        typ = metric_type(metric)
        adverse_steps = {}
        for method in df_common['method'].unique():
            series = df_common[df_common['method'] == method][metric].reset_index(drop=True)
            if len(series) < 2:
                continue
            diff = series.diff().fillna(0)
            adverse_count = (diff > 0).sum() if typ == 'cost' else (diff < 0).sum()
            adverse_steps[method] = int(adverse_count)
        if not adverse_steps:
            continue
        min_adv = min(adverse_steps.values())
        best_methods = [m for m, v in adverse_steps.items() if v == min_adv]
        report_lines.append(f"{metric}: {', '.join(best_methods)} ({min_adv} adverse variations)")

    txt_path = os.path.join(comparison_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"\nReport saved at: {txt_path}")

    # -------- Gráficos de linha: Top-3 + Estabilidade --------
    for metric in metric_list:
        if metric not in df_common.columns:
            continue

        # Top-3 por pódio (desempenho)
        podium_points = podium_score_df[metric]
        top3 = podium_points.sort_values(ascending=False).index[:3].tolist()
        ranks = {m: i + 1 for i, m in enumerate(top3)}

        # Vencedor por janela (para dots vermelhos)
        best_per_window = find_best_per_window(df_common, metric)

        # Pontos de estabilidade (3/2/1)
        stability_points = compute_stability_points(df_common, metric)
        max_stab = max(stability_points.values()) if stability_points else 0
        most_stable_methods = [m for m, pts in stability_points.items() if pts == max_stab]

        # Preparar figura
        plt.figure(figsize=(14, 7))
        colors = sns.color_palette("muted", 3)
        handles, legend_labels = [], []
        plotted_methods = set()

        # 1) Top-3 (sólido) e, se também for “Most Stable”, fica tracejado grosso
        for method in top3:
            dat = df_common[df_common['method'] == method]
            x = dat['step']
            y = dat[metric]
            rank = ranks[method]
            is_stable = method in most_stable_methods

            if is_stable:
                label = f"{method} ({rank}{'st' if rank==1 else 'nd' if rank==2 else 'rd'} & Most Stable)"
                line, = plt.plot(x, y, label=label, linewidth=2.8,
                                 color=colors[(rank - 1) % 3], linestyle="--", alpha=1)
            else:
                label = f"{method} ({rank}{'st' if rank==1 else 'nd' if rank==2 else 'rd'})"
                line, = plt.plot(x, y, label=label, linewidth=2.3,
                                 color=colors[(rank - 1) % 3])
            handles.append(line)
            legend_labels.append(label)
            plotted_methods.add(method)

        # 2) “Most Stable” que não está no top‑3 (tracejado azul)
        for method in most_stable_methods:
            if method in plotted_methods:
                continue
            dat = df_common[df_common['method'] == method]
            x = dat['step']; y = dat[metric]
            label = f"{method} (Most Stable)"
            line, = plt.plot(x, y, label=label, linewidth=2.8,
                             color='blue', linestyle="--", alpha=0.9)
            handles.append(line)
            legend_labels.append(label)
            plotted_methods.add(method)

        # 3) Restante em cinza
        for method in df_common['method'].unique():
            if method in plotted_methods:
                continue
            dat = df_common[df_common['method'] == method]
            plt.plot(dat['step'], dat[metric], color='gray', alpha=0.18, linewidth=1)

        # 4) Dots vermelhos (vencedores fora dos destacados) com texto abaixo
        for _, row in best_per_window.iterrows():
            if row['method'] in plotted_methods:
                continue
            folder, meth = (row['method'].split('/', 1) + [''])[:2] if '/' in row['method'] else ('', row['method'])
            plt.scatter(row['step'], row['best_value'], color='red', marker='o', s=70, zorder=10)
            full_label = f"{folder}\n{meth}" if folder else meth
            gap = 0.04
            plt.text(row['step'], row['best_value'] - gap * max(1, abs(row['best_value'])),
                     full_label, color='red', fontsize=8, va='top', ha='center',
                     fontweight='bold', linespacing=1.3)

        # Rótulos
        plt.xlabel('Window (step)')
        plt.ylabel(metric.replace('_', ' ').capitalize())

        # ======= LEGENDA EM UMA ÚNICA LINHA (duas soluções combinadas) =======
        # (A) Forçar todos na mesma linha:
        ncols = max(1, len(legend_labels))
        # (B) Permitir que o box se alongue horizontalmente (sem aumentar a figura),
        #     com pequenos ajustes de espaçamento:
        plt.legend(
            handles=handles,
            labels=legend_labels,
            frameon=True,
            ncol=ncols,                    # força 1 linha
            bbox_to_anchor=(0.5, 1.05),    # move o box ligeiramente para cima e permite largura
            loc='lower center',
            borderaxespad=0.2,
            labelspacing=0.5,
            handletextpad=0.6
        )

        # Forçar fonte nos elementos
        ax = plt.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(prop)
        ax.xaxis.label.set_fontproperties(prop)
        ax.yaxis.label.set_fontproperties(prop)
        if ax.legend_ is not None:
            for text in ax.legend_.get_texts():
                text.set_fontproperties(prop)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        outpdf = os.path.join(comparison_dir, f'{metric}_top3_stability_lines.pdf')
        plt.savefig(outpdf, bbox_inches='tight')
        plt.close()
        print(f'Top-3/stability podium plot saved at {outpdf}')
