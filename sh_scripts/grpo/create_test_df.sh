#!/bin/bash

# Konfiguracja ścieżek (relatywne do miejsca uruchomienia)
DATA_DIR="Agents/data/datasets"
ORIG_TRAIN="$DATA_DIR/rl_grounded_dataset_train.csv"
ORIG_EVAL="$DATA_DIR/rl_grounded_dataset_test.csv"

NEW_TRAIN="$DATA_DIR/rl_grounded_dataset_train_80.csv"
NEW_EVAL="$DATA_DIR/rl_grounded_dataset_test_20.csv"

echo "=> Tworzenie małych zbiorów danych (Train: 80, Eval: 20)..."

# 1. Tworzenie zbioru TRAIN
# Pobieramy nagłówek
head -n 1 "$ORIG_TRAIN" > "$NEW_TRAIN"
# Pomijamy nagłówek (tail +2), mieszamy (shuf) i bierzemy 80 wierszy
tail -n +2 "$ORIG_TRAIN" | shuf -n 80 >> "$NEW_TRAIN"

# 2. Tworzenie zbioru EVAL
# Pobieramy nagłówek
head -n 1 "$ORIG_EVAL" > "$NEW_EVAL"
# Pomijamy nagłówek, mieszamy i bierzemy 20 wierszy
tail -n +2 "$ORIG_EVAL" | shuf -n 20 >> "$NEW_EVAL"

echo "=> Gotowe!"
echo "   Nowy Train: $NEW_TRAIN"
echo "   Nowy Eval:  $NEW_EVAL"