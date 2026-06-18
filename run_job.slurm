#!/bin/bash
#SBATCH --account=e33188
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=qwen_vlm_lab_v4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "================================================================"
echo "V4 Lab Notebook Parser — SLURM Job"
echo "Started on $(date)"
echo "Node: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
echo "================================================================"

# -------------------------
# Move to project directory
# -------------------------
cd /projects/e33188/praneeth/labally

mkdir -p logs
mkdir -p lab_parser_outputs_v4

# -------------------------
# Load environment
# -------------------------
module purge
module load mamba/24.3.0

eval "$(conda shell.bash hook)"
conda activate /gpfs/projects/e33188/praneeth/labally/myenv

# -------------------------
# Cache paths (use project storage, not home quota)
# -------------------------
export HF_HOME=/projects/e33188/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=/projects/e33188/torch_cache
export TMPDIR=/projects/e33188/tmp/${SLURM_JOB_ID}

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TMPDIR"

# GPU memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU thread settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------------
# Install V4 dependencies (idempotent; skips already-installed packages)
# -------------------------
echo ""
echo "--- Installing V4 dependencies ---"

pip install --quiet --upgrade \
    "pix2tex[cli]" \
    molscribe \
    "DECIMER>=2.4" \
    chemdataextractor2 \
    rdkit \
    scipy

# Install latest transformers (needed for TrOCR + Qwen2.5-VL)
pip install --quiet --upgrade --no-deps \
    git+https://github.com/huggingface/transformers

echo "Dependencies installed."

# -------------------------
# Diagnostics
# -------------------------
echo ""
echo "--- Diagnostics ---"
echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print("GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# -------------------------
# Run V4 parser
# -------------------------
echo ""
echo "--- Running V4 Parser ---"
IMAGE_PATH="/projects/e33188/praneeth/labally/Example_lab_notebook_page.jpg"

python run_parser.py \
  --image "$IMAGE_PATH" \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --output-dir "lab_parser_outputs_v4" \
  --max-width 1600 \
  --backend full \
  --pubchem \
  --visualize

echo ""
echo "--- Parser complete ---"

# -------------------------
# Validate output
# -------------------------
echo ""
echo "--- Running validation checks ---"
OUTPUT_JSON=$(ls lab_parser_outputs_v4/Example_lab_notebook_page_v4_full.json 2>/dev/null | head -1 || \
              ls lab_parser_outputs_v4/*.json 2>/dev/null | head -1 || echo "")

if [ -n "$OUTPUT_JSON" ]; then
    python tests/validate_output.py "$OUTPUT_JSON" --verbose
else
    echo "WARNING: No output JSON found to validate."
fi

echo ""
echo "================================================================"
echo "Job finished on $(date)"
echo "================================================================"