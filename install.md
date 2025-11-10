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

L3
```
bash scripts/train/math/math_L3_fresh.sh
```

Example generation setup
```
bash scripts/train/code/code_L3_example.sh
```

