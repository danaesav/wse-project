import os

import pandas as pd
from transformers import TrainerCallback


class ClientCSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path, round_num, client_id):
        self.csv_path = csv_path
        self.round_num = round_num
        self.client_id = client_id
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs_filtered = {
                "round": self.round_num,
                "client": self.client_id,
                "epoch": state.epoch,
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "accuracy": logs.get("accuracy"),
                "f1": logs.get("f1")
            }
            self.logs.append(logs_filtered)

    def on_train_end(self, args, state, control, **kwargs):
        if self.logs:
            df = pd.DataFrame(self.logs)
            if os.path.exists(self.csv_path):
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, index=False)


def log_global_metrics(round_num, metrics, csv_path):
    row = {"round": round_num}
    row.update({k: metrics[k] for k in ["eval_loss", "accuracy", "f1"] if k in metrics})
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def save_total_time(seconds_elapsed, filepath):
    with open(filepath, 'w') as f:
        f.write(f"Total training time: {seconds_elapsed:.2f} seconds\n")


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        f.write(f"Final metrics on test set: \n {metrics} \n")
