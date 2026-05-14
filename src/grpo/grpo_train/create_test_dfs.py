import pandas as pd
import os
import sys

# Dynamiczne ustalanie ścieżki bazowej (zakładamy uruchomienie z root projektu)
# Ale dla bezpieczeństwa używamy ścieżek relatywnych do struktury Agents/
BASE_DATA_DIR = "Agents/data/datasets"

orig_train = os.path.join(BASE_DATA_DIR, "rl_grounded_dataset_train.csv")
orig_eval = os.path.join(BASE_DATA_DIR, "rl_grounded_dataset_test.csv")

new_train = os.path.join(BASE_DATA_DIR, "rl_grounded_dataset_train_80.csv")
new_eval = os.path.join(BASE_DATA_DIR, "rl_grounded_dataset_test_20.csv")

def create_subsets():
    print(f" Checking for files in: {BASE_DATA_DIR}")
    
    if not os.path.exists(orig_train) or not os.path.exists(orig_eval):
        print(f"ERROR: Could not find original files in {BASE_DATA_DIR}")
        sys.exit(1)

    # Procesowanie Train (80)
    print("=> Processing Train subset (80 samples)...")
    df_train = pd.read_csv(orig_train)
    df_train_sub = df_train.sample(n=80, random_state=42)
    df_train_sub.to_csv(new_train, index=False)

    # Procesowanie Eval (20)
    print("=> Processing Eval subset (20 samples)...")
    df_eval = pd.read_csv(orig_eval)
    df_eval_sub = df_eval.sample(n=20, random_state=42)
    df_eval_sub.to_csv(new_eval, index=False)

    print(f"\n=> SUCCESS!")
    print(f"   Created: {new_train}")
    print(f"   Created: {new_eval}")

if __name__ == "__main__":
    create_subsets()