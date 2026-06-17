# ms-swift Server Environment

This repo includes a reusable setup script for `ms-swift` on the server.

## Create or update the env

```bash
cd /data3/weiyiyang/code/SciFullMMBench
bash tests/scripts/setup_ms_swift_env.sh
```

## Activate the env

```bash
source /data3/weiyiyang/anaconda3/etc/profile.d/conda.sh
conda activate /data3/weiyiyang/code/SciFullMMBench/.conda-ms-swift
```

## Smoke tests

```bash
swift --help
python -c "import swift, torch; print(swift.__version__); print(torch.cuda.is_available())"
```

## Notes

- The setup script installs CUDA-enabled PyTorch from the `cu124` wheel index.
- `ms-swift` is installed with `pip install -U ms-swift`.
- The current repo is also installed editable into the same env.
