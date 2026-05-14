import pandas as pd
import os

# Ścieżki
data_dir = "Agents/data/datasets"
orig_train = os.path.join(data_dir, "rl_grounded_dataset_train.csv")
orig_eval = os.path.join(data_dir, "rl_grounded_dataset_test.csv")

new_train = os.path.join(data_dir, "rl_grounded_dataset_train_80.csv")
new_eval = os.path.join(data_dir, "rl_grounded_dataset_test_20.csv")

print("=> Tworzenie małych zbiorów danych za pomocą Pandas...")

# Używamy pandas, bo on rozumie, że tekst w cudzysłowie to jedna komórka, 
# nawet jeśli ma w środku entery.
df_train = pd.read_csv(orig_train)
df_train.sample(n=80, random_state=42).to_csv(new_train, index=False)

df_eval = pd.read_csv(orig_eval)
df_eval.sample(n=20, random_state=42).to_csv(new_eval, index=False)

print(f"=> Gotowe!\n   Train: {new_train}\n   Eval: {new_eval}")