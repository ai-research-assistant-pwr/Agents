#!/bin/bash
#SBATCH -p lem-cpu-short
#SBATCH --job-name=weaviate_test
#SBATCH --output=out/weaviate_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

BASE_DIR=$(pwd -P)

echo "Uruchamianie Weaviate (metoda RUN)..."

# Używamy dokładnie tych samych flag, które zadziałały w srun
apptainer run --contain \
    --no-home \
    --bind "$BASE_DIR/weaviate_data:/var/lib/weaviate" \
    --env "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true" \
    --env "PERSISTENCE_DATA_PATH=/var/lib/weaviate" \
    --env "CLUSTER_HOSTNAME=localhost" \
    weaviate.sif &

# Zapisujemy PID procesu, żeby SLURM go widział
WEAVIATE_PID=$!

echo "Czekam 30 sekund na start..."
sleep 30

# Test lokalny (wewnątrz skryptu sbatch)
echo "Testowanie połączenia lokalnie na węźle..."
curl -s http://localhost:8080/v1/meta

# Trzymamy zadanie przy życiu
wait $WEAVIATE_PID
