#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -J rl_selection
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

cd /home/yeojinjung/rl-synthetic-data
eval "$(~/bin/micromamba shell hook -s bash)"
micromamba activate rl-synth
export HF_TOKEN=""
export PYTHONPATH=.

#echo ">>> Preparing Data..."
#python -u scripts/0_prepare_data.py

echo ">>> Creating swapped candidate pools..."
python -u scripts/swap_female.py --in data/processed/candidate_pool.jsonl --seed 0

BASE_POOL="data/processed/candidate_pool.jsonl"
POOLS=(
  "candidate_pool"
  "candidate_pool_0.6"
  "candidate_pool_0.99"
)

for POOL_NAME in "${POOLS[@]}"; do
  echo "=============================="
  echo ">>> Running pipeline for ${POOL_NAME}"
  echo "=============================="

  POOL_JSONL="data/processed/${POOL_NAME}.jsonl"
  if [[ ! -f "${POOL_JSONL}" ]]; then
    echo "Missing ${POOL_JSONL}, skipping ${POOL_NAME}"
    continue
  fi

  # Gradients (per pool)
  if [[ -f "data/features/${POOL_NAME}/pool_grads.pt" && -f "data/features/${POOL_NAME}/val_grads.pt" ]]; then
    echo ">>> Gradients already exist for ${POOL_NAME}, skipping."
  else
    echo ">>> Extracting Gradients..."
    python -u scripts/1_extract_gradients.py \
      --pool-jsonl "${POOL_JSONL}" \
      --val-jsonl "data/processed/val_set.jsonl" \
      --out-dir "data/features/${POOL_NAME}"
  fi

  # Selection
  if ls "data/selected_indices/${POOL_NAME}/"*_indices.npy >/dev/null 2>&1; then
    echo ">>> Selected indices already exist for ${POOL_NAME}, skipping."
  else
    echo ">>> Running Selection Algorithms..."
    time python -u scripts/2_selectors.py \
      --k 2000 \
      --pool-path "${POOL_JSONL}" \
      --val-path "data/processed/val_set.jsonl" \
      --pool-grads "data/features/${POOL_NAME}/pool_grads.pt" \
      --val-grads "data/features/${POOL_NAME}/val_grads.pt" \
      --out-dir "data/selected_indices/${POOL_NAME}"
  fi

  # Selection outputs already written to pool-specific dir

  # Training (SEQUENTIAL)
  echo ">>> Fine-tuning Llama-3.2-1B..."
  for METHOD in less dsir prismatic_soft random
  do
    echo ">>> Starting training for: $METHOD"
    time python -u scripts/3_train_student.py --method "$METHOD" --pool-jsonl "${POOL_JSONL}" --pool-name "${POOL_NAME}"|| echo "Training failed for $METHOD"
  done

  # Move models to pool-specific dir
  mkdir -p "models/${POOL_NAME}"
  for METHOD in less dsir prismatic_soft random
  do
    if [[ -d "models/final-${METHOD}" ]]; then
      mv "models/final-${METHOD}" "models/${POOL_NAME}/final-${METHOD}"
    fi
  done

  ## UMAP Visualizations
  #echo ">>> Running UMAP Visualizations..."
  #time python -u scripts/5_visualize.py \
  #  --pool-path "${POOL_JSONL}" \
  #  --pool-grads "data/features/${POOL_NAME}/pool_grads.pt" \
  #  --val-grads "data/features/${POOL_NAME}/val_grads.pt" \
  #  --indices-dir "data/selected_indices/${POOL_NAME}" \
  #  --out-dir "outputs/${POOL_NAME}/umap"

  # StereoSet Bias Evaluation
  echo ">>> Running StereoSet bias evaluation..."
  time python -u scripts/6_eval_stereoset.py \
    --models-dir "models/${POOL_NAME}" \
    --out-dir "outputs/${POOL_NAME}/stereo"

  # ID/OOD Evaluation + log-prob outputs
  echo ">>> Running ID/OOD Evaluation..."
  time python -u scripts/4_evaluate.py \
    --pool-name "${POOL_NAME}" \
    --indices-dir "data/selected_indices/${POOL_NAME}" \
    --models-dir "models/${POOL_NAME}" \
    --pool-grads "data/features/${POOL_NAME}/pool_grads.pt" \
    --val-grads "data/features/${POOL_NAME}/val_grads.pt" \
    --pool-jsonl "${POOL_JSONL}" \
    --max-examples 200
done

echo ">>> Experiment Complete."
