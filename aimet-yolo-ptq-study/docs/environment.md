# Environment Notes

## Host Layout

Expected host setup:

```text
Windows host
  WSL2 Ubuntu
    Docker Engine
    NVIDIA Container Toolkit
    AIMET official/prebuilt ONNX GPU container
```

## Required Checks

Run these from WSL2 Ubuntu before starting experiments:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Then start the AIMET container:

```bash
bash docker/build_runtime_image.sh
bash docker/run_aimet_onnx_gpu.sh
```

Inside the AIMET container:

```bash
python -m pip install -r requirements.txt
python docker/verify_aimet.py
python scripts/00_check_env.py
```

## AIMET Image

The runtime image in `docker/run_aimet_onnx_gpu.sh` is:

```text
aimet-yolo-onnx-gpu:2.2.0
```

It is built from the prebuilt AIMET development image documented by AIMET:

```text
artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:latest.onnx-gpu
```

The prebuilt development image contains the ONNX/CUDA dependency stack, but AIMET itself is installed from the official 2.2.0 ONNX GPU wheel during `docker/build_runtime_image.sh`.

If the tag changes, run with:

```bash
AIMET_IMAGE="<official-image>:<tag>" bash docker/run_aimet_onnx_gpu.sh
```

## Version Capture

Each experiment should record:

- AIMET version
- ONNX Runtime version
- ONNX version
- CUDA-visible provider list
- Docker image name and tag
- GPU name and driver version
- Model file hash
- Dataset annotation file hash

## Project Dependencies

The official AIMET image supplies AIMET itself. The project-level `requirements.txt` adds the study utilities:

- `ultralytics` for YOLO26 model export and baseline validation
- `pycocotools` for COCO metrics workflows
- `PyYAML` for reading project configs
- `requests` and `tqdm` for robust downloads
