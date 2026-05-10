#!/bin/bash
# Colab / Kaggle setup: replaces upstream setup.sh (which needs `pixi`).
# Run from the repo root:  bash eval/setup_colab.sh
set -euo pipefail

echo "==> Installing Python deps via pip"
pip install -q --upgrade pip
pip install -q \
    numpy==1.26.4 \
    pandas tqdm gdown \
    requests beautifulsoup4 lxml \
    scikit-learn scipy \
    matplotlib pillow \
    tldextract unidecode psutil \
    fvcore pycocotools \
    opencv-python opencv-contrib-python \
    flask flask-cors \
    pyyaml

echo "==> Installing PyTorch (uses Colab/Kaggle's preinstalled CUDA)"
# Colab/Kaggle ship a recent torch already; skip if present.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || pip install -q torch torchvision

echo "==> Installing Detectron2"
# Build from source against the active torch. Slow (~5 min) but version-safe.
pip install -q --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "==> Downloading Phishpedia models"
mkdir -p models
cd models

declare -A FILES=(
    ["rcnn_bet365.pth"]="1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH"
    ["faster_rcnn.yaml"]="1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW"
    ["resnetv2_rgb_new.pth.tar"]="1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS"
    ["expand_targetlist.zip"]="1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I"
    ["domain_map.pkl"]="1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1"
)

for fname in "${!FILES[@]}"; do
    if [ ! -s "$fname" ]; then
        echo "  -> $fname"
        gdown --id "${FILES[$fname]}" -O "$fname"
    else
        echo "  [skip] $fname"
    fi
done

echo "==> Extracting expand_targetlist.zip"
if [ ! -d "expand_targetlist" ] || [ -z "$(ls -A expand_targetlist 2>/dev/null)" ]; then
    unzip -oq expand_targetlist.zip -d expand_targetlist
    # Flatten if zip created a nested expand_targetlist/expand_targetlist/
    if [ -d "expand_targetlist/expand_targetlist" ]; then
        mv expand_targetlist/expand_targetlist/* expand_targetlist/
        rmdir expand_targetlist/expand_targetlist
    fi
fi

cd ..
echo "==> Done. Verify with:  python -c 'from configs import load_config; load_config()'"
