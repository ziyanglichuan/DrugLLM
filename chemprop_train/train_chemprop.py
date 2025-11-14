import os
import pandas as pd
import subprocess
import shutil
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import time
import json


data_dir = "./assay_activity_data"
model_dir = "./model"
pred_dir = "./preds"
best_model_dir = "./best_model"
log_dir = "./train_logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

pearson_target = 0.75


def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    if df.empty:
        return None, None, None

    # Automatically apply log10 transform
    is_wide_range = df["standard_value"].max() / max(df["standard_value"].min(), 1e-9) > 100
    is_not_normalized = df["standard_value"].min() >= 0 or df["standard_value"].max() > 2

    if is_wide_range and is_not_normalized:
        print(f"INFO: Applying log10 transform to {os.path.basename(file_path)}.")
        df["standard_value"] = np.log10(df["standard_value"].clip(lower=1e-9))

    scaler = StandardScaler()
    df["standard_value"] = scaler.fit_transform(df[["standard_value"]])
    return df, scaler, True


def run_cmd(cmd, assay_name):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{assay_name} command failed: {e.stderr}")
        print(f"Failed command: {' '.join(cmd)}")
        return None


def find_model_files(model_dir_path):
    """
    Find model_0, model_1, ... files inside model_dir_path
    """
    if model_dir_path is None or not os.path.exists(model_dir_path):
        return []

    model_files = []
    i = 0
    while True:
        model_file = os.path.join(model_dir_path, f"model_{i}")
        if os.path.exists(model_file):
            model_files.append(model_file)
            i += 1
        else:
            break

    return model_files


def train_and_eval(assay_name, df, scaler, round_id, config):
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=round_id * 42)
    temp_train = f"./temp_train_{assay_name}.csv"
    temp_test = f"./temp_test_{assay_name}.csv"
    train_df.to_csv(temp_train, index=False)
    test_df.to_csv(temp_test, index=False)

    model_path = os.path.join(model_dir, f"{assay_name}_round{round_id}")
    preds_path = os.path.join(pred_dir, f"{assay_name}_round{round_id}_preds.csv")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    # Build training command
    train_cmd = [
        "chemprop", "train",
        "--data-path", temp_train,
        "--save-dir", model_path,
        "--smiles-columns", "canonical_smiles",
        "--target-columns", "standard_value",
        "--split-type", "scaffold_balanced",
        "--epochs", str(config["epochs"]),
        "--patience", str(config["patience"]),
        "--message-hidden-dim", str(config["hidden_size"]),
        "--ffn-hidden-dim", str(config["hidden_size"]),
        "--dropout", str(config["dropout"]),
        "--metrics", "r2",
        "--accelerator", "gpu",
        "--devices", "1"
    ]

    if config["features"]:
        train_cmd += ["--features-generators", "v1_rdkit_2d_normalized"]

    if config["ensemble"] > 1:
        train_cmd += ["--ensemble-size", str(config["ensemble"])]

    if config.get("checkpoint"):
        found_checkpoints = find_model_files(config["checkpoint"])
        if found_checkpoints:
            train_cmd += ["--checkpoint"] + found_checkpoints
            print(f"INFO: {assay_name} Round {round_id} loading {len(found_checkpoints)} checkpoints from {config['checkpoint']}")
        else:
            print(f"{assay_name} Round {round_id}: Checkpoint not found; training from scratch. (Searching 'model_i' inside {config['checkpoint']})")

    # Train
    print(f"Round {round_id} training {assay_name}...")
    out = run_cmd(train_cmd, assay_name)
    if not out:
        for f in [temp_train, temp_test]:
            if os.path.exists(f): os.remove(f)
        return None, model_path

    # Find trained model files
    best_model_files = find_model_files(model_path)

    if not best_model_files:
        print(f"{assay_name} Round {round_id} training succeeded but no model_i files found!")
        print(f"    (Search path: {model_path})")
        for f in [temp_train, temp_test]:
            if os.path.exists(f): os.remove(f)
        return None, model_path

    print(f"INFO: {assay_name} Round {round_id} found {len(best_model_files)} models for prediction.")

    # Prediction
    predict_cmd = [
        "chemprop", "predict",
        "--test-path", temp_test,
        "--model-path",
    ] + best_model_files + [
        "--preds-path", preds_path,
        "--accelerator", "gpu",
        "--devices", "1"
    ]

    if config["features"]:
        predict_cmd += ["--features-generators", "v1_rdkit_2d_normalized"]

    out = run_cmd(predict_cmd, assay_name)
    if not out:
        for f in [temp_train, temp_test]:
            if os.path.exists(f): os.remove(f)
        return None, model_path

    try:
        preds = pd.read_csv(preds_path)
    except pd.errors.EmptyDataError:
        print(f"{assay_name} Round {round_id} prediction failed: prediction file is empty: {preds_path}")
        for f in [temp_train, temp_test]:
            if os.path.exists(f): os.remove(f)
        return None, model_path

    y_pred = preds["standard_value"].values
    y_true = test_df["standard_value"].values

    if y_pred.shape[0] != y_true.shape[0]:
        print(f"{assay_name} Round {round_id}: prediction size ({y_pred.shape[0]}) does not match ground truth ({y_true.shape[0]}).")
        return None, model_path
    if y_pred.shape[0] == 0:
        print(f"{assay_name} Round {round_id}: no valid predictions.")
        return None, model_path

    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        print(f"{assay_name} Round {round_id}: std of predictions or ground truth is near zero; Pearson r cannot be computed.")
        r = 0.0
    else:
        r, _ = pearsonr(y_true, y_pred)

    for f in [temp_train, temp_test, preds_path]:
        if os.path.exists(f):
            os.remove(f)

    return r, model_path


def get_adaptive_config(round_id, best_model_round):
    """
    Generate configuration based on global round and best model round.
    """

    configs = {
        1: {"epochs": 50, "patience": 5, "hidden_size": 600, "dropout": 0.1, "ensemble": 1, "features": False},
        2: {"epochs": 80, "patience": 8, "hidden_size": 800, "dropout": 0.1, "ensemble": 1, "features": True},
        3: {"epochs": 100, "patience": 10, "hidden_size": 1000, "dropout": 0.15, "ensemble": 3, "features": True},
        4: {"epochs": 120, "patience": 12, "hidden_size": 1200, "dropout": 0.2, "ensemble": 5, "features": True},
        5: {"epochs": 150, "patience": 15, "hidden_size": 1500, "dropout": 0.25, "ensemble": 5, "features": True},
        6: {"epochs": 200, "patience": 20, "hidden_size": 1800, "dropout": 0.3, "ensemble": 5, "features": True},
    }

    if round_id <= 6:
        config_to_use = configs[round_id].copy()
    else:
        config_to_use = configs[6].copy()

    config_to_use["checkpoint"] = None

    if best_model_round is not None:
        best_model_config = configs.get(best_model_round, configs[6])

        arch_match = (
            config_to_use["hidden_size"] == best_model_config["hidden_size"]
            and config_to_use["features"] == best_model_config["features"]
        )

        if arch_match:
            config_to_use["checkpoint"] = "LOAD_PREVIOUS_BEST"
        else:
            print(f"INFO: Round {round_id} (Config {config_to_use['hidden_size']}) does not match best model architecture (Round {best_model_round}, Config {best_model_config['hidden_size']}). Training from scratch.")

    return config_to_use



print("Initialization started...")
assay_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(f"Found {len(assay_files)} assays. Preloading data...")

final_results = []
active_assays = {}

for assay_file in assay_files:
    assay_name = assay_file[:-4]

    final_path = os.path.join(best_model_dir, f"{assay_name}_best")
    info_path = os.path.join(final_path, "info.json")

    if os.path.exists(final_path):
        saved_r = np.nan
        status_msg = "Success (Loaded from disk)"

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                    saved_r = info_data.get("best_r", np.nan)

                if not np.isnan(saved_r):
                    status_msg = f"Success (Loaded r={saved_r:.4f})"
                else:
                    status_msg = "Success (Loaded, r missing)"
            except Exception as e:
                print(f"Could not read info.json for {assay_name}: {e}")
                status_msg = "Success (Loaded, info.json corrupted)"
        else:
            print(f"INFO: Old model found for {assay_name} (info.json missing), r set to NaN.")

        print(f"{assay_name} already exists in {best_model_dir}, skipping. (R={saved_r})")
        final_results.append((assay_name, status_msg, saved_r))
        continue

    try:
        df, scaler, ok = load_and_preprocess(os.path.join(data_dir, assay_file))
        if not ok:
            print(f"{assay_name} data invalid or empty, skipping.")
            final_results.append((assay_name, "Skipped (No data)", np.nan))
            continue

        active_assays[assay_name] = {
            "df": df,
            "scaler": scaler,
            "best_r": -1,
            "best_model_path": None,
            "best_model_round": None,
        }
        print(f"   -> {assay_name} loaded successfully.")
    except Exception as e:
        print(f"Failed to load {assay_name}: {e}, skipping.")
        final_results.append((assay_name, f"Skipped (Load Error: {e})", np.nan))

print(f"Initialization completed. {len(active_assays)} active assays, {len(final_results)} skipped.")

# BFS infinite loop
round_id = 0
while len(active_assays) > 0:
    round_id += 1
    print("\n" + "="*40)
    print(f"🌀 Starting global Round {round_id} ({len(active_assays)} assays remaining)")
    print("="*40)

    current_active_list = list(active_assays.keys())

    for i, assay_name in enumerate(current_active_list):
        print(f"\n--- [ Assay {i+1}/{len(current_active_list)} (Global Round {round_id}) ] ---")
        print(f"Processing Assay: {assay_name}")

        state = active_assays[assay_name]

        config = get_adaptive_config(round_id, state["best_model_round"])

        if config["checkpoint"] == "LOAD_PREVIOUS_BEST":
            config["checkpoint"] = state["best_model_path"]

        r, new_model_path = train_and_eval(
            assay_name,
            state["df"],
            state["scaler"],
            round_id,
            config
        )

        previous_best_path = state["best_model_path"]

        if r is None:
            print(f"{assay_name} Round {round_id} failed.")
            if new_model_path and os.path.exists(new_model_path):
                shutil.rmtree(new_model_path)
            continue

        print(f"{assay_name} Round {round_id} Pearson r = {r:.4f}")

        if r > state["best_r"]:
            state["best_r"] = r
            state["best_model_path"] = new_model_path
            state["best_model_round"] = round_id
            print(f"New best score for {assay_name} (best r={state['best_r']:.4f} @ Round {round_id}).")

            if previous_best_path and os.path.exists(previous_best_path) and previous_best_path != new_model_path:
                print(f"Cleaning old best model for {assay_name}: {os.path.basename(previous_best_path)}")
                shutil.rmtree(previous_best_path)
        else:
            print(f"{assay_name} Round {round_id} did not surpass best score ({state['best_r']:.4f}).")
            if new_model_path and os.path.exists(new_model_path):
                print(f"Removing current round model: {os.path.basename(new_model_path)}")
                shutil.rmtree(new_model_path)

        if state["best_r"] >= pearson_target:
            print(f"{assay_name} reached target (best r={state['best_r']:.4f})! Saving model and removing from loop.")

            final_path = os.path.join(best_model_dir, f"{assay_name}_best")
            if os.path.exists(final_path):
                shutil.rmtree(final_path)

            shutil.copytree(state["best_model_path"], final_path)
            joblib.dump(state["scaler"], os.path.join(final_path, "scaler.pkl"))

            info_data = {
                "assay_name": assay_name,
                "best_r": state["best_r"],
                "best_model_round": state["best_model_round"]
            }
            info_path = os.path.join(final_path, "info.json")
            try:
                with open(info_path, 'w', encoding='utf-8') as f:
                    json.dump(info_data, f, indent=4)
            except Exception as e:
                print(f"Critical Error: Failed to write info.json to {final_path}: {e}")

            print(f"Best model for {assay_name} saved (r={state['best_r']:.4f}): {final_path}")

            final_results.append((assay_name, "Success", state["best_r"]))

            print(f"🧹 Cleaning remaining intermediate model for {assay_name}...")
            if state["best_model_path"] and os.path.exists(state["best_model_path"]):
                shutil.rmtree(state["best_model_path"])

            del active_assays[assay_name]

        else:
            print(f"{assay_name} Round {round_id} not yet reaching target (best r={state['best_r']:.4f}), retry in next round...")

    print("\n" + "-"*30)
    print(f"Round {round_id} finished. Updating final_results.csv...")

    current_summary = []
    current_summary.extend(final_results)

    for assay_name, state in active_assays.items():
        status = f"Pending (best r={state['best_r']:.4f})"
        current_summary.append((assay_name, status, state["best_r"]))

    summary_df = pd.DataFrame(current_summary, columns=["Assay", "Status", "Best_r"])
    summary_df = summary_df.sort_values(by="Best_r", ascending=False, na_position='last')
    summary_df.to_csv("final_results.csv", index=False)

    print(f"{len(active_assays)} assays still active.")
    print("-" * 30)

print("\n" + "="*30)
print(f"All tasks completed!")
print(summary_df)
print("="*30)
