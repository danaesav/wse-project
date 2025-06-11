import os
import pandas as pd
import matplotlib.pyplot as plt


root_dir = 'results'

name_mapping = {
    'FedAvg_b01' : 'FedAvg (b=0.1)',
    'FedAvg_b05' : 'FedAvg (b=0.5)',
    'FedAvg_b09' : 'FedAvg (b=0.9)',
    'FedAvg_uniform' : 'FedAvg (uniform)',
}

plt.figure(figsize=(10, 6))

summary_data = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        label = os.path.basename(subdir)
        if file == 'global_eval_log.csv':
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)

            if 'round' in df.columns and 'eval_loss' in df.columns:
                plt.plot(df['round'], df['eval_loss'], label=name_mapping[label])
        if file == 'final_logs.csv':
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)
            row = df.iloc[0].to_dict()
            row["Model"] = name_mapping[label]
            summary_data.append(row)


plt.xlabel('Round')
plt.ylabel('Eval Loss')
plt.title('Evaluation Loss over Rounds')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/fedavg_eval_loss_plot.png')
plt.show()

summary_df = pd.DataFrame(summary_data)
summary_df.set_index("Model", inplace=True)
summary_df = summary_df[[
    "eval_loss", "eval_accuracy", "eval_f1", "total_time_seconds"
]]

print(summary_df)