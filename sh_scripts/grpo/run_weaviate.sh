#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH --job-name=weaviate_test
#SBATCH --output=out/weaviate_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

BASE_DIR=$(pwd -P)

# Fresh directory for Weaviate's runtime data (must be empty/new)
mkdir -p "$BASE_DIR/data/weaviate_runtime"

# The backup directory contains: backup_config.json + node1/
# It is mounted as /var/backups/research-paper-embed-qwen-4b inside the container
# so the backup_id = "research-paper-embed-qwen-4b"
BACKUP_ID="research-paper-embed-qwen-4b"

echo "Uruchamianie Weaviate (metoda RUN) in $BASE_DIR..."

apptainer run --contain \
    --no-home \
    --bind "$BASE_DIR/data/weaviate_runtime:/var/lib/weaviate" \
    --bind "$BASE_DIR/data/weaviate_backup:/var/backups" \
    --env "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true" \
    --env "PERSISTENCE_DATA_PATH=/var/lib/weaviate" \
    --env "CLUSTER_HOSTNAME=node1" \
    --env "BACKUP_FILESYSTEM_PATH=/var/backups" \
    --env "ENABLE_MODULES=backup-filesystem" \
    data/weaviate.sif &

WEAVIATE_PID=$!

echo "Czekam 30 sekund na start..."
sleep 30

echo "Testowanie połączenia lokalnie na węźle..."
curl -s http://localhost:8080/v1/meta
echo ""

# Restore backup only on first run (runtime dir is empty / has no node data yet)
if [ ! -d "$BASE_DIR/data/weaviate_runtime/researchpapers" ]; then
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
    echo "Dane juz istnieja w weaviate_runtime/ - pomijam restore."
fi

echo "Sprawdzanie schematu..."
curl -s http://localhost:8080/v1/schema
echo ""

# Trzymamy zadanie przy życiu
wait $WEAVIATE_PID
