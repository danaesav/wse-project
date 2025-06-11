import time
from typing import List, Union
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import Trainer, TrainingArguments
from types import MethodType
import copy
import torch

from CSVLogger import log_global_metrics, ClientCSVLoggerCallback, save_total_time, save_metrics
from utils import compute_metrics
import os  # Imported but not explicitly used in the provided snippet. Keep for potential future use.

# Assuming compute_metrics is available from utils.py
# Example placeholder if utils.py is not provided:
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# def compute_metrics(p):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = preds.argmax(axis=1)
#     if hasattr(p.label_ids, 'cpu'):
#         labels = p.label_ids.cpu().numpy()
#     else:
#         labels = p.label_ids
#     precision, recall, f1, _ = precision_recall_fscore_support(labels,
#                                                                preds, average='binary') # or 'weighted' based on task
#     acc = accuracy_score(labels, preds)
#     return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def get_model_copy(model):
    """
    Creates a deep copy of the input PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be copied.

    Returns:
        torch.nn.Module: A deep copy of the model.
    """
    return copy.deepcopy(model)


def aggregate_models(global_model, client_models, client_weights):
    """
    Aggregates the state dictionaries of client models into the global model
    using weighted averaging.

    Args:
        global_model (torch.nn.Module): The global model to be updated.
        client_models (List[torch.nn.Module]): A list of client models.
        client_weights (List[float]): A list of weights corresponding to each client model.
                                     These weights should sum to 1.
    """
    global_dict = global_model.state_dict()

    for k in global_dict:
        global_dict[k] = torch.zeros_like(global_dict[k])

    for client_model, weight in zip(client_models, client_weights):
        client_dict = client_model.state_dict()
        for k in global_dict:
            global_dict[k] += client_dict[k].to(global_dict[k].device) * weight

    # Load the aggregated state dictionary back into the global model
    global_model.load_state_dict(global_dict)


def federated_train(
        base_model,
        client_datasets: List[Union[DataLoader, HFDataset]],
        val_ds: HFDataset,
        test_ds: HFDataset,
        client_weights: List[float] = None,
        local_epochs: int = 1,
        global_rounds: int = 3,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        run_name: str = None
):
    """
    Implements a federated learning training loop.

    Args:
        base_model (torch.nn.Module): The initial model architecture.
        client_datasets (List[Union[DataLoader, HFDataset]]): A list of datasets or DataLoaders,
                                                              one for each client.
        val_ds (HFDataset): The validation dataset for evaluating client models (optional)
                            and global model during training.
        test_ds (HFDataset): The test dataset for final evaluation of the global model.
        client_weights (List[float], optional): Weights for each client in aggregation.
                                                If None, equal weights are used. Defaults to None.
        local_epochs (int, optional): Number of local epochs for each client. Defaults to 1.
        global_rounds (int, optional): Number of global aggregation rounds. Defaults to 3.
        batch_size (int, optional): Batch size for training and evaluation. Defaults to 32.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 5e-5.
        device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        run_name: str: Name of the run. Used for logging and creating a path. Defaults to None.

    Returns:
        torch.nn.Module: The final global model after all federated rounds.
    """
    client_log_path = f"results/{run_name}/client_training_log.csv"
    global_log_path = f"results/{run_name}/global_eval_log.csv"
    final_log_path = f"results/{run_name}/final_logs.txt"
    os.makedirs(f"results/{run_name}", exist_ok=True)
    total_start_time = time.time()
    num_clients = len(client_datasets)

    # Assertions for client weights
    assert len(client_weights) == num_clients, "Mismatch between client_weights and number of clients"
    assert abs(sum(client_weights) - 1.0) < 1e-5, "Client weights must sum to 1"

    # Initialize the global model and move to the specified device
    global_model = get_model_copy(base_model).to(device)

    # Common evaluation arguments for global model evaluation
    # Set logging and saving strategy to "no" to prevent unnecessary file creation
    evaluation_args = TrainingArguments(
        output_dir="./results/global_eval_temp",  # Temporary output directory for evaluation
        per_device_eval_batch_size=batch_size,
        logging_strategy="no",
        save_strategy="no",
        disable_tqdm=True,  # Disable progress bar for cleaner output
        report_to="none",  # Do not report to any logging service
    )

    for round_idx in range(global_rounds):
        print(f"\n--- Global Round {round_idx + 1}/{global_rounds} ---")

        # Evaluate the global model before client updates in each round
        print(" Evaluating global model before client updates...")
        evaluator = Trainer(
            model=global_model,  # Use the current global model for evaluation
            args=evaluation_args,
            eval_dataset=test_ds,  # Evaluate on the test dataset
            compute_metrics=compute_metrics,
        )
        metrics = evaluator.evaluate()
        print(f" Global Evaluation (Round {round_idx + 1}): {metrics}")
        log_global_metrics(round_num=round_idx + 1, metrics=metrics, csv_path=global_log_path)

        # === Train on each client ===
        client_models = []
        for client_idx, client_data in enumerate(client_datasets):
            print(f" Training on Client {client_idx + 1}/{num_clients}...")

            # Get a copy of the current global model for client training
            client_model = get_model_copy(global_model).to(device)

            # Define training arguments for the client
            # Ensure unique output_dir for each client and round
            training_args = TrainingArguments(
                # output_dir=f"./results/client_{client_idx}_round_{round_idx}",
                num_train_epochs=local_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,  # For evaluation during client training
                logging_strategy="epoch",  # Log metrics at the end of each epoch
                save_strategy="no",  # Do not save client models
                learning_rate=learning_rate,
                weight_decay=0.01,
                disable_tqdm=True,
                report_to="none",
                load_best_model_at_end=False,  # We don't need to load the best model as we aggregate
            )

            logger_cb = ClientCSVLoggerCallback(
                csv_path=client_log_path,
                round_num=round_idx + 1,
                client_id=client_idx + 1
            )

            # Initialize Trainer based on whether client_data is a DataLoader or HFDataset
            if isinstance(client_data, DataLoader):
                trainer = Trainer(
                    model=client_model,
                    args=training_args,
                    # eval_dataset can be None or val_ds based on need for client-side validation
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                    callbacks=[logger_cb],
                )
                # Override get_train_dataloader for DataLoader input
                trainer.get_train_dataloader = MethodType(lambda self: client_data, trainer)
            else:  # Assuming it's an HFDataset
                trainer = Trainer(
                    model=client_model,
                    args=training_args,
                    train_dataset=client_data,
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                    callbacks=[logger_cb],
                )

            # Perform local training for the client
            trainer.train()
            # Move client model to CPU before appending to avoid potential device issues during aggregation
            client_models.append(client_model.cpu())

        # === Aggregate updates ===
        print(" Aggregating client models into global model...")
        aggregate_models(global_model, client_models, client_weights)
        # Move the global model back to the device for the next round's evaluation/client distribution
        global_model.to(device)

    # === Final Evaluation after all rounds ===
    print("\n--- Final Evaluation on Test Set ---")
    final_trainer = Trainer(
        model=global_model.to(device),  # Ensure global model is on the correct device for final eval
        args=evaluation_args,  # Reuse evaluation arguments
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    final_metrics = final_trainer.evaluate()
    print(f" Final Test Metrics: {final_metrics}")

    total_time = time.time() - total_start_time
    print(f"Total time for {global_rounds} rounds: {total_time:.2f} seconds")
    save_total_time(total_time, final_log_path)
    save_metrics(final_metrics, final_log_path)

    return global_model
