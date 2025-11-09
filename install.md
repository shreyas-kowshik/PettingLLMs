conda create -n petlm python=3.12
conda activate petlm
git submodule update --init --recursive
pip install uv
cd verl
uv pip install -e .
cd ..

uv pip install --upgrade "torch" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

# Install ninja
uv pip install -U ninja

# Install flash-attn from source, needed for Babel looks like
cd # Or to any custom path to clone flash-attention for installing from source
git clone https://github.com/Dao-AILab/flash-attention.git
uv pip install --no-build-isolation .

cd /path/to/PettingLLMS
uv pip install -r "requirements_venv.txt"


### V2

`bash setup.bash`

```
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

```

### Get stuff working
```
source "pettingllms_venv/bin/activate"
```

```
python scripts/dataprocess/load_math.py
```

```
export HF_HOME="/data/user_data/skowshik/huggingface"
export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"
bash scripts/train/math/math_L1_prompt.sh
```

