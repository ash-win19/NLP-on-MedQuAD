#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="medquad_env"

echo "==> Creating/refreshing conda env: ${ENV_NAME} with Python 3.10"
# Use Mamba if available for faster solving, otherwise fall back to conda
if command -v mamba &> /dev/null; then
    INSTALLER="mamba"
else
    INSTALLER="conda"
fi

# Create env if it doesn't exist
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${INSTALLER}" create -n "${ENV_NAME}" python=3.10 -y
fi

# Always activate
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Installing core scientific stack & PyTorch via conda channels"
# Use conda/pytorch channel for PyTorch, which is generally better at resolving
# the low-level libomp/MKL conflicts on Apple Silicon.
"${INSTALLER}" install -y -c conda-forge -c pytorch \
  numpy \
  pandas \
  scikit-learn \
  matplotlib \
  seaborn \
  pytorch \
  jupyterlab \
  lxml \
  pyarrow

echo "==> Installing NLP/IR stack & Metrics via pip"
pip install \
  spacy \
  transformers \
  datasets \
  evaluate \
  rank-bm25 \
  faiss-cpu \
  sentence-transformers \
  rouge-score \
  bert-score \
  sacrebleu \
  ipykernel \
  tqdm \
  rich

echo "==> Downloading spaCy English Model"
python -m spacy download en_core_web_sm

echo "==> Registering Jupyter kernel"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "==> Done. Activate the env with: conda activate ${ENV_NAME}"
echo "==> You can now start Jupyter Lab/Notebook from this terminal."