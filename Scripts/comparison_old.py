import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ajuste para imports do projeto
target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(target_path)

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

def plot_metric(df, metric, agg=None, ylabel=None, save_dir='Comparison', figsize=(10,5)):
    plt.figure(figsize=figsize)
    if agg is not None:
        field = f"{metric}_{agg}"
        if field not in df.columns:
            if metric in df.columns:
                field = metric
            else:
                raise KeyError(f"Coluna {field} e {metric} não encontradas no DataFrame!")
    else:
        if metric not in df.columns:
            raise KeyError(f"Coluna {metric} não encontrada no DataFrame!")
        field = metric

    for method in sorted(df['method'].unique()):
        sel = df[df['method'] == method]
        y = sel[field]
        x = sel['step']
        plt.plot(x, y, marker='o', label=method)
    plt.xlabel('Janela (step)')
    plt.ylabel(ylabel or metric)
    title_field = field if agg is not None else metric
    plt.title(f"{title_field.replace('_', ' ').capitalize()} por método")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{title_field}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico salvo em: {plot_path}")

def find_best_per_window(df, metric_field):
    best_rows = []
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if not candidates.empty:
            idxmin = candidates[metric_field].idxmin()
            best = candidates.loc[idxmin]
            best_rows.append({
                'step': step,
                'method': best['method'],
                'best_value': best[metric_field]
            })
    return pd.DataFrame(best_rows)

def podium_ranking(df, metric_field):
    points = {}
    for step, group in df.groupby('step'):
        candidates = group.dropna(subset=[metric_field])
        if candidates.empty:
            continue
        sorted_group = candidates.sort_values(metric_field)
        podium = sorted_group['method'].values[:3]
        for i, pts in enumerate([3,2,1]):
            if i < len(podium):
                method = podium[i]
                points[method] = points.get(method, 0) + pts
    return points

if __name__ == "__main__":
    # ----- CONFIGURÁVEL -----
    SCRIPT_DIR = "Paper/"
    method_dirs = [
        os.path.join(SCRIPT_DIR, "PPO_MLP"),
        os.path.join(SCRIPT_DIR, "PPO_MHA"),
        # Adicione mais se quiser comparar outras pastas
    ]
    comparison_dir = os.path.join(SCRIPT_DIR, "Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    df = load_all_results_multi(method_dirs)
    csv_path = os.path.join(comparison_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tabela consolidada salva em: {csv_path}")

    # Gráficos principais
    plot_metric(df, 'sequential_costs', agg='mean', ylabel='Sequential Cost (média)', save_dir=comparison_dir)
    plot_metric(df, 'sequential_rewards', agg='mean', ylabel='Sequential Reward (média)', save_dir=comparison_dir)
    plot_metric(df, 'standard_total_cost', ylabel='Total Cost', save_dir=comparison_dir)
    plot_metric(df, 'standard_total_reward', ylabel='Total Reward', save_dir=comparison_dir)

    # ==== GRÁFICOS EXTRAS ====
    metric_list = [
        'sequential_costs_mean',
        'sequential_rewards_mean',
        'standard_total_cost',
        'standard_total_reward'
    ]

    # Boxplot por janela (Step)
    for metric in metric_list:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(14,6))
        sns.boxplot(x='step', y=metric, data=df)
        plt.xlabel('Janela (step)')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f'Distribuição de {metric.replace("_", " ")} por janela')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{metric}_boxplot_step.png'), dpi=300)
        plt.close()

    # Boxplot por método
    for metric in metric_list:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(14,6))
        sns.boxplot(x='method', y=metric, data=df)
        plt.xticks(rotation=45)
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f'Distribuição de {metric.replace("_", " ")} por método')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{metric}_boxplot_method.png'), dpi=300)
        plt.close()

    # Barra: valor médio por método
    for metric in metric_list:
        if metric not in df.columns:
            continue
        means = df.groupby('method')[metric].mean().sort_values()
        plt.figure(figsize=(12,6))
        means.plot(kind='bar')
        plt.ylabel(f'{metric.replace("_", " ").capitalize()} (média por método)')
        plt.title(f'Média de {metric.replace("_", " ")} por método')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{metric}_bar_mean_method.png'), dpi=300)
        plt.close()

    # Heatmap método vs step
    for metric in metric_list:
        if metric not in df.columns:
            continue
        pivot = df.pivot(index='method', columns='step', values=metric)
        plt.figure(figsize=(18,7))
        sns.heatmap(pivot, annot=False, cmap='viridis')
        plt.xlabel('Janela (step)')
        plt.ylabel('Método')
        plt.title(f'Heatmap de {metric.replace("_", " ")} por método e janela')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{metric}_heatmap.png'), dpi=300)
        plt.close()

    # Gráfico de linhas Top-3 métodos (média)
    for metric in metric_list:
        if metric not in df.columns:
            continue
        means = df.groupby('method')[metric].mean().sort_values()
        top3 = means.index[:3]
        plt.figure(figsize=(12,6))
        for method in df['method'].unique():
            dat = df[df['method']==method]
            if method in top3:
                plt.plot(dat['step'], dat[metric], label=method, linewidth=2)
            else:
                plt.plot(dat['step'], dat[metric], color='gray', alpha=0.3)
        plt.xlabel('Janela (step)')
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.title(f'Evolução dos Top 3 métodos em {metric.replace("_", " ")}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'{metric}_top3_lines.png'), dpi=300)
        plt.close()

    # --------- RELATÓRIO TXT ----------
    report_lines = []
    report_lines.append(f"Tabela consolidada salva em: {csv_path}\n")

    metrics_to_check = metric_list
    summary = {}

    for metric in metrics_to_check:
        if metric not in df.columns:
            msg = f"Coluna {metric} não existe nos resultados, pulando."
            print(msg)
            report_lines.append(msg)
            continue
        best_df = find_best_per_window(df, metric)
        score = best_df['method'].value_counts().sort_values(ascending=False)
        summary[metric] = score
        out_csv = os.path.join(comparison_dir, f'best_{metric}_per_window.csv')
        best_df.to_csv(out_csv, index=False)
        msg1 = f"\nMelhores modelos por janela para {metric} (salvo em {out_csv}):"
        msg2 = best_df.head().to_string(index=False)
        print(msg1)
        print(msg2)
        report_lines.append(msg1)
        report_lines.append(msg2)

        detailed = "\n".join(
            f"Janela {row.step}: {row.method} (valor={row.best_value:.4f})"
            for row in best_df.itertuples()
        )
        print("\nRanking por janela:")
        print(detailed)
        report_lines.append("\nRanking por janela:")
        report_lines.append(detailed)

        msg3 = "\nPontuação (maior é melhor):"
        msg4 = score.to_string()
        print(msg3)
        print(msg4)
        report_lines.append(msg3)
        report_lines.append(msg4)

        if not score.empty:
            max_score = score.max()
            winners = score[score == max_score].index.tolist()
            winner_msg = f"Winner(s): {', '.join(winners)} [score={max_score}]"
            print(winner_msg)
            report_lines.append(winner_msg)

    all_methods = sorted(df['method'].unique())
    all_metrics = metrics_to_check
    full_score_df = pd.DataFrame(0, index=all_methods, columns=all_metrics)
    for metric, vc in summary.items():
        for method, v in vc.items():
            full_score_df.loc[method, metric] = v
    score_full_path = os.path.join(comparison_dir, 'score_full_table.csv')
    full_score_df.to_csv(score_full_path)
    print(f"\nTabela geral de scores salva em {score_full_path}")

    podium_summary = {}
    for metric in metrics_to_check:
        if metric not in df.columns:
            continue
        points = podium_ranking(df, metric)
        podium_summary[metric] = points

    podium_score_df = pd.DataFrame(0, index=all_methods, columns=metrics_to_check)
    for metric, dct in podium_summary.items():
        for method, pts in dct.items():
            podium_score_df.loc[method, metric] = pts
    podium_path = os.path.join(comparison_dir, 'podium_score_full_table.csv')
    podium_score_df.to_csv(podium_path)
    print(f"\nTabela geral de pontos (pódio) salva em {podium_path}")

    report_lines.append("\nINTERPRETAÇÃO DO RELATÓRIO")
    report_lines.append("- Para cada métrica avaliada, o método vencedor em cada janela é aquele que obteve o MENOR valor da métrica ('menor é melhor' para a métrica).")
    report_lines.append("- O ranking clássico mostra o número de vezes em que cada método foi 1º lugar ('maior é melhor' para score).")
    report_lines.append("- O ranking de pódio soma pontos: 3 para 1º lugar, 2 para 2º, 1 para 3º, 0 para os demais, em cada janela. O total reflete o desempenho consistente nas primeiras posições.\n")
    report_lines.append("TABELA GERAL DE SCORES (número de janelas vencidas por método e métrica):\n")
    report_lines.append(full_score_df.to_string())

    report_lines.append("\nTABELA GERAL DE PONTOS POR PÓDIO (número de pontos por método e métrica):\n")
    report_lines.append(podium_score_df.to_string())
    for metric in metrics_to_check:
        if metric not in podium_score_df.columns:
            continue
        podium_col = podium_score_df[metric]
        max_pts = podium_col.max()
        winners = podium_col[podium_col == max_pts].index.tolist()
        winner_msg = f"Winner(s) (pódio) para {metric}: {', '.join(winners)} [score={max_pts}]"
        print(winner_msg)
        report_lines.append(winner_msg)

    txt_path = os.path.join(comparison_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"\nRelatório salvo em: {txt_path}")
