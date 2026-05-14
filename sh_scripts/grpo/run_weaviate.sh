#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH --job-name=weaviate_test
#SBATCH --output=Agents/out/weaviate_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Ustalamy bazowy katalog projektu na Agents
MY_DISK="$SLURM_SUBMIT_DIR"
BASE_DIR="$MY_DISK/Agents"

# Katalog na dane bazy (runtime) i backup znajdują się u Ciebie w Agents/data/
RUNTIME_DIR="$BASE_DIR/data/weaviate_runtime"
BACKUP_DIR="$BASE_DIR/data/weaviate_backup"

# Tworzymy folder na runtime, jeśli go nie ma
mkdir -p "$RUNTIME_DIR"

# ID backupu musi zgadzać się z nazwą folderu wewnątrz weaviate_backup
BACKUP_ID="research-paper-embed-qwen-4b"

echo "Uruchamianie Weaviate..."
echo "Lokalny Runtime: $RUNTIME_DIR"
echo "Lokalny Backup: $BACKUP_DIR"

apptainer run --contain \
    --no-home \
    --bind "$RUNTIME_DIR:/var/lib/weaviate" \
    --bind "$BACKUP_DIR:/var/backups" \
    --env "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true" \
    --env "PERSISTENCE_DATA_PATH=/var/lib/weaviate" \
    --env "CLUSTER_HOSTNAME=node1" \
    --env "BACKUP_FILESYSTEM_PATH=/var/backups" \
    --env "ENABLE_MODULES=backup-filesystem" \
    "$BASE_DIR/data/weaviate.sif" &

WEAVIATE_PID=$!

echo "Czekam 30 sekund na start..."
sleep 30

echo "Testowanie połączenia lokalnie na węźle..."
curl -s http://localhost:8080/v1/meta
echo ""

# Jeśli kolekcja nie istnieje w runtime, wykonujemy restore z lokalnego backupu
if [ ! -d "$RUNTIME_DIR/researchpapers" ]; then
    echo "Pierwszy start - przywracanie backupu: $BACKUP_ID ..."
    curl -s -X POST "http://localhost:8080/v1/backups/filesystem/$BACKUP_ID/restore" \
        -H "Content-Type: application/json" \
        -d '{}'
    echo ""

    echo "Czekam 120 sekund na restore..."
    sleep 120

    echo "Sprawdzanie statusu restore..."
    curl -s "http://localhost:8080/v1/backups/filesystem/$BACKUP_ID/restore"
    echo ""
else
    echo "Dane juz istnieja w Twoim weaviate_runtime/ - pomijam restore."
fi

echo "Sprawdzanie schematu..."
curl -s http://localhost:8080/v1/schema
echo ""

# Trzymamy zadanie przy życiu
wait $WEAVIATE_PID