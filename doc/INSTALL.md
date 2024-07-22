# Install Virtual Env

**Step 1: create conda virtual environment**

```bash
conda create -n fsd python=3.8
conda activate fsd
```

**Step 2: install cuda compiler**
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

**Step 3: install torch**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: install openmm libraries**

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

For `mmdet`, we need build on local to utilize its development feature

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

