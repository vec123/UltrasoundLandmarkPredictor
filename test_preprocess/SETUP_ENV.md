# Setup instructions for preprocessing environment

## Option A: Use existing `medim_env` (recommended)

If you already have `medim_env` set up, just install the missing package:

```powershell
& "C:\Users\vic-b\Documents\Victors\Projects\environments\medim_env\Scripts\Activate.ps1"
cd "C:\Users\vic-b\Documents\Victors\Projects\fetusnet"
pip install pynrrd
```

## Option B: Create a fresh environment

Create a new Python environment dedicated to preprocessing (no GPU/torch needed):

```powershell
# Create conda environment with Python 3.13
conda create -n fetusnet_preprocess python=3.13 -y

# Activate it
conda activate fetusnet_preprocess

# Navigate to fetusnet folder
cd "C:\Users\vic-b\Documents\Victors\Projects\fetusnet"

# Install requirements
pip install -r test_preprocess/requirements.txt
```

## Verify the setup

Check that the required packages are installed:

```powershell
python -c "import numpy, scipy, SimpleITK, pynrrd, yaml; print('✓ All imports successful')"
```

## Run preprocessing

From `C:\Users\vic-b\Documents\Victors\Projects\fetusnet`, execute:

```powershell
python fetusnet.py  --mode=preprocess  --ds.sys="C:\Users\vic-b\Documents\Victors\Projects\fetusnet\" --ds.root="test_preprocess" --ds.dataframe="dataframe" --preprocessing.save_dir="test_preprocess/" --preprocessing.filter=True  --preprocessing.bspline=True --preprocessing.affine=False
```

The output will be in `test_preprocess/volumes/1-20s-01.nrrd`.
