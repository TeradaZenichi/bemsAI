import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Caminhos dos arquivos JSON
paths = {
    "EWC": "costs_rewards_log - EWC.json",
    "SI": "costs_rewards_log - SI.json",
    "MAS": "costs_rewards_log - MAS.json",
    "None": "costs_rewards_log - None.json",
}

# 2. Leitura dos arquivos e extração dos dados principais
plot_data_full = {}
for name, path in paths.items():
    with open(path, "r") as f:
        log = json.load(f)
        x = [entry["train_days"][0] for entry in log][1:]
        seq = [entry["sequential_costs"][0] for entry in log][1:]
        seq_r = [entry["sequential_rewards"][0] for entry in log][1:]
        std = [entry["standard_costs_mean"] for entry in log][1:]
        total_cost = [entry["standard_total_cost"] for entry in log][1:]
        total_reward = [entry["standard_total_reward"] for entry in log][1:]
        plot_data_full[name] = {
            "x": x,
            "seq": seq,
            "seq_r": seq_r,
            "std": std,
            "total_cost": total_cost,
            "total_reward": total_reward,
        }

# 3. Encontrar os dias (ciclos) comuns entre todos os métodos
all_x = [set(plot_data_full[m]["x"]) for m in plot_data_full.keys()]
common_days = sorted(list(set.intersection(*all_x)))
methods = list(plot_data_full.keys())
labels = common_days

# 4. Função utilitária para pegar vetor alinhado de cada método/campo
def get_aligned(met, field):
    return [plot_data_full[met][field][plot_data_full[met]["x"].index(day)] for day in common_days]

# 5. (Opcional) Gerar CSV com todos os dados alinhados
aligned_data = {
    "day": common_days
}
for m in methods:
    aligned_data[f"{m}_sequential_costs"] = get_aligned(m, "seq")
    aligned_data[f"{m}_sequential_rewards"] = get_aligned(m, "seq_r")
    aligned_data[f"{m}_standard_total_cost"] = get_aligned(m, "total_cost")
    aligned_data[f"{m}_standard_total_reward"] = get_aligned(m, "total_reward")

df = pd.DataFrame(aligned_data)
df.to_csv("aligned_continual_rl_results.csv", index=False)
print("CSV salvo: aligned_continual_rl_results.csv")
print(df.head())

# 6. Plot: Sequential Cost
plt.figure(figsize=(12,5))
for i, method in enumerate(methods):
    plt.bar(np.arange(len(labels)) + i*0.2 - 0.3, get_aligned(method, "seq"), width=0.2, label=method)
plt.xticks(np.arange(len(labels)), labels)
plt.xlabel("First Training Day of Window")
plt.ylabel("Sequential Cost")
plt.title("Sequential Cost per Training Window (Grouped Bars, from Day 2, aligned)")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Plot: Sequential Reward
plt.figure(figsize=(12,5))
for i, method in enumerate(methods):
    plt.bar(np.arange(len(labels)) + i*0.2 - 0.3, get_aligned(method, "seq_r"), width=0.2, label=method)
plt.xticks(np.arange(len(labels)), labels)
plt.xlabel("First Training Day of Window")
plt.ylabel("Sequential Reward")
plt.title("Sequential Reward per Training Window (Grouped Bars, from Day 2, aligned)")
plt.legend()
plt.tight_layout()
plt.show()

# 8. Plot: Standard Total Cost
plt.figure(figsize=(12,5))
for i, method in enumerate(methods):
    plt.bar(np.arange(len(labels)) + i*0.2 - 0.3, get_aligned(method, "total_cost"), width=0.2, label=method)
plt.xticks(np.arange(len(labels)), labels)
plt.xlabel("First Training Day of Window")
plt.ylabel("Standard Total Cost")
plt.title("Standard Total Cost per Training Window (Grouped Bars, from Day 2, aligned)")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Plot: Standard Total Reward
plt.figure(figsize=(12,5))
for i, method in enumerate(methods):
    plt.bar(np.arange(len(labels)) + i*0.2 - 0.3, get_aligned(method, "total_reward"), width=0.2, label=method)
plt.xticks(np.arange(len(labels)), labels)
plt.xlabel("First Training Day of Window")
plt.ylabel("Standard Total Reward")
plt.title("Standard Total Reward per Training Window (Grouped Bars, from Day 2, aligned)")
plt.legend()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

methods = list(plot_data_full.keys())
labels = ["Sequential Cost", "Sequential Reward", "Standard Total Cost", "Standard Total Reward"]

# Função para pegar diferença do dia 2 - dia 1 para cada método/campo
def get_delta(field):
    return [get_aligned(m, field)[1] - get_aligned(m, field)[0] for m in methods]

# Coletar os deltas
deltas = [
    get_delta("seq"),
    get_delta("seq_r"),
    get_delta("total_cost"),
    get_delta("total_reward"),
]

# Gráfico de barras agrupadas dos deltas
x = np.arange(len(labels))
width = 0.18

plt.figure(figsize=(10,5))
for i, method in enumerate(methods):
    plt.bar(x + i*width - width*1.5, [d[i] for d in deltas], width, label=method)
plt.xticks(x, labels, rotation=20)
plt.ylabel("Δ (Day 2 - Day 1)")
plt.title("Variation from First to Second Common Day (Grouped Bars)")
plt.legend()
plt.tight_layout()
plt.show()
