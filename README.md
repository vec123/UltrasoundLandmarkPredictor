# FetusNet

FetusNet is a research codebase for automatic facial landmark detection in fetal 3D ultrasound volumes. The main pipeline trains a 3D ResUNet-style network to predict heatmaps for 19 anatomical facial landmarks and evaluates localization error in millimetres.

The project was developed at Universitat Pompeu Fabra / DTIC for fetal facial 3D ultrasound landmark detection experiments.

## Main features

- 3D ultrasound preprocessing with optional filtering, B-spline resampling, axis swap, and affine standardization.
- Heatmap-based target generation for 19 fetal facial landmarks.
- 3D ResUNet training with configurable loss functions.
- Evaluation using mean landmark localization error and expected local accuracy curves.
- Optional Weights & Biases logging.
- Optional 3D visualization / interactive modes for predictions and landmarks.

## Repository structure

```text
fetusnet.py                  # Main entry point for all modes
net/config/default.yaml      # Default configuration
net/config/parser.py         # CLI override parser
net/model/                   # Network architectures and model utilities
net/loss/                    # Loss functions
net/phases/                  # Training and inference loops
net/evaluation/              # Evaluation metrics
dataset/                     # Dataset, preprocessing, target generation, split utilities
dataset/statistics/          # Metadata preparation and cross-fold splitting
dataset/utility/             # Image I/O, rotations, standardization utilities
assets/                      # Figures and visual assets
scripts/                     # Plotting / figure-generation scripts
doc/requirements.txt         # Python dependencies
templates/                   # NRRD header template used during evaluation output saving
```

## Installation

The code has been developed with Python 3.8 and CUDA-enabled PyTorch. A typical setup is:

```bash
conda create -n fetusnet python=3.8 -y
conda activate fetusnet

# Choose the PyTorch build that matches your CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r doc/requirements.txt
```

Check that PyTorch can see your GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Configuration and command-line usage

All modes are launched through `fetusnet.py` (except some of the scripts for figure creation). Runtime options come from `net/config/default.yaml` and can be overridden from the command line using dotted keys:

```bash
python fetusnet.py --mode=train --training.epochs=100 --training.batch_size=2
```

Lists must be quoted so the shell passes them as one argument:

```bash
python fetusnet.py --training.loss="['softmaxce', 'eucEMD']" --training.lambdas="[1, 0.1]"
```

Useful modes include:

| Mode | Purpose |
|---|---|
| `prepare` | Scan raw datasets and create metadata CSV files. |
| `presplit` | Create grouped cross-fold train/validation/test CSV files. |
| `preprocess` | Preprocess volumes and transform landmarks. |
| `train` | Train the model and save checkpoints. |
| `test` | Evaluate a saved checkpoint. |
| `generate` | Generate target volumes for selected patients. |
| `interactive_plot` | Visualize volumes, landmarks, and outputs. |
| `interactive_game` | Launch the interactive landmark game. |
| `test_loaders` | Sanity-check the data loaders. |

You can inspect all current default values with:

```bash
python fetusnet.py --help
```

## Expected data layout

The code expects a system root and a dataset root:

```text
<ds.sys>/<ds.root>/
├── <dataframe>.csv
├── volumes/
│   └── <nsid>.nrrd
└── landmarks/
    ├── csv/
    │   └── <nsid>.csv
    └── fcsv/
        └── <nsid>.fcsv
```

The main dataframe is selected by:

```yaml
ds:
  sys: "/path/to/system/root/"
  root: "DATA/"
  dataframe: "sinfo_fold0"
```

so the script loads:

```text
/path/to/system/root/DATA/sinfo_fold0.csv
```

For training/testing, the dataframe should contain at least the columns used by the loaders and filters:

| Column | Meaning |
|---|---|
| `set` | Split assignment: `0=train`, `1=validation`, `2=test`. |
| `nsid` | Scan identifier. |
| `npid` | Patient identifier. |
| `mscan` | Relative path to the processed `.nrrd` volume. |
| `mcsv` | Relative path to the processed landmark `.csv`. |
| `visibles` | Visible landmark labels for visibility-aware evaluation. |
| `nonfrontal_after_rot` | Boolean flag used to remove non-frontal scans. |

For preprocessing, the dataframe also needs raw-data columns such as `fscan`, `_fcsv`, `fcaso`, `osid`, `mlmk`, and `mcsv`, which are produced by the preparation utilities.

## Example workflow

### 1. Prepare metadata from raw data

Preparation scans the raw dataset folders, checks image/landmark availability, and writes metadata CSV files such as `sinfo_total.csv`, `pinfo_total.csv`, and `sinfo.csv`.

```bash
python fetusnet.py \
  --mode=prepare \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.source="" \
  --ds.dataframe="sinfo" \
  --split.train_val_ds="['Estudio Dexeus']" \
  --split.test_ds="['Casos Mar']"
```

Notes:

- Raw dataset folder conventions are currently hard-coded in `dataset/statistics/prepare.py`.
- The standard-plane dictionary is read from `doc/info/gt.txt`.
- If your dataframe does not already contain a `set` column, run the split step before training.

### 2. Create train/validation/test splits

The split utility assigns `set=0` to training scans, `set=1` to validation scans, and `set=2` to test scans. It uses grouped folds by patient ID (`npid`).

```bash
python fetusnet.py \
  --mode=presplit \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.dataframe="sinfo" \
  --split.n=4 \
  --split.train_val_ds="['Estudio Dexeus']" \
  --split.test_ds="['Casos Mar']" \
  --test_patients="[]"
```

This creates files like:

```text
/media/.../sinfo_fold0.csv
/media/.../sinfo_fold1.csv
/media/.../sinfo_fold2.csv
/media/.../sinfo_fold3.csv
```

### 3. Preprocess images and landmarks

The preprocessing mode loads raw volumes and raw landmark CSV/FCSV files, then saves standardized volumes and transformed landmarks under `DATA/volumes/` and `DATA/landmarks/`.

```bash
python fetusnet.py --config default.yaml --nrrd my_nrrd.nrrd



python fetusnet.py \
  --mode=preprocess \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.dataframe="sinfo_fold0" \
  --preprocessing.save_dir="DATA/" \
  --preprocessing.filter=True \
  --preprocessing.bspline=True \
  --preprocessing.swap=True \
  --preprocessing.affine=True \
  --preprocessing.gtpp=False \
  --preprocessing.params.filter.filter_size=3 \
  --preprocessing.params.bspline.spacing="[1.0, 1.0, 1.0]" \
  --preprocessing.params.bspline.size="[128, 128, 128]"
```

Expected outputs:

```text
/media/.../DATA/volumes/<nsid>.nrrd
/media/.../DATA/landmarks/csv/<nsid>.csv
/media/.../DATA/landmarks/fcsv/<nsid>.fcsv
```

### 4. Sanity-check the data loaders

Before launching a long training run, check that the dataframe, volumes, and landmarks are readable:

```bash
python fetusnet.py \
  --mode=test_loaders \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.dataframe="sinfo_fold0" \
  --training.batch_size=2 \
  --validation.batch_size=1
```

### 5. Train

```bash
python fetusnet.py \
  --mode=train \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.dataframe="sinfo_fold0" \
  --prefix="resunet3d_fold0" \
  --device="cuda" \
  --training.architecture="resunet3d" \
  --training.epochs=50 \
  --training.batch_size=2 \
  --validation.batch_size=1 \
  --training.optimizer="adam" \
  --optimizer.adam.lr=0.0001 \
  --training.loss="['softmaxce']" \
  --training.lambdas="[1]" \
  --training.num_fts=32 \
  --wandb.log=False \
  --validation.use_model="last"
```

Training creates an experiment directory under:

```text
runs/YYYY-MM-DD/<prefix>/
```

Typical outputs include:

```text
runs/YYYY-MM-DD/<prefix>/params.yaml
runs/YYYY-MM-DD/<prefix>/last.pt
runs/YYYY-MM-DD/<prefix>/best.pt
runs/YYYY-MM-DD/<prefix>/losses_<timestamp>.csv
runs/YYYY-MM-DD/<prefix>/test_scores.csv
runs/YYYY-MM-DD/<prefix>/test_scores_mean.csv
```

> Current code note: `fetusnet.py` references `params.validation.use_model` after training. Until the config is made consistent, pass `--validation.use_model="last"` or `--validation.use_model="best"` in train mode.

### 6. Test a saved checkpoint

```bash
python fetusnet.py \
  --mode=test \
  --checkpoint="runs/2026-07-07/resunet3d_fold0" \
  --validation.use_model="best" \
  --ds.sys="/media/.../" \
  --ds.root="DATA/" \
  --ds.dataframe="sinfo_fold0" \
  --device="cuda" \
  --test_patients="[1002, 1003]" \
  --validation.detector="argmax" \
  --validation.radius_eval=100 \
  --validation.radius_num=100 \
  --validation.save_outputs=True \
  --validation.save_targets=True \
  --validation.show_figures=False
```

If the selected `test_patients` are not found, the script falls back to the full test subset. Evaluation results are saved in the checkpoint directory, and per-landmark output/target NRRDs are saved under:

```text
<checkpoint>/eval/
```

## Losses and model options

Supported architecture:

| Value | Description |
|---|---|
| `resunet3d` | 3D residual U-Net backbone. |

Supported losses:

| Value | Description |
|---|---|
| `sse` | Squared error loss. |
| `softmaxce` | Softmax cross-entropy style heatmap loss. |
| `eucEMD` | Euclidean distance / EMD-inspired regularized loss. |

Example with multiple losses:

```bash
python fetusnet.py \
  --mode=train \
  --ds.dataframe="sinfo_fold0" \
  --training.loss="['softmaxce', 'eucEMD']" \
  --training.lambdas="[1.0, 0.1]" \
  --training.mnl=False \
  --validation.use_model="best"
```

## Weights & Biases

Enable W&B logging with:

```bash
python fetusnet.py \
  --mode=train \
  --wandb.log=True \
  --wandb.wandbpro="fetusnetv3" \
  --ds.dataframe="sinfo_fold0" \
  --validation.use_model="best"
```

You must be logged in before running:

```bash
wandb login
```

## Troubleshooting

### `Master dataframe not found`

Check that the following file exists:

```text
<ds.sys>/<ds.root>/<ds.dataframe>.csv
```

For example, with `--ds.sys="/media/.../"`, `--ds.root="DATA/"`, and `--ds.dataframe="sinfo_fold0"`, the script expects:

```text
/media/.../DATA/sinfo_fold0.csv
```

### `KeyError: 'set'`

The dataframe must contain a `set` column before training/testing. Use `presplit` or add the split column manually:

```text
0 = train
1 = validation
2 = test
```

### `KeyError: 'nonfrontal_after_rot'`

`utils.update_dataframe()` filters out non-frontal scans using the `nonfrontal_after_rot` column. Add this column to the dataframe, or update the filtering logic if your dataset does not use this flag.

### Checkpoint not found

For test mode, `--checkpoint` must point to an experiment directory containing `best.pt` or `last.pt`:

```text
runs/YYYY-MM-DD/<prefix>/best.pt
runs/YYYY-MM-DD/<prefix>/last.pt
```

Select the checkpoint using:

```bash
--validation.use_model="best"
```

or:

```bash
--validation.use_model="last"
```

## Citation

**If you use in any academic work, please wait until we have a DOI number, so you can cite.**

## License

See [`LICENSE`](LICENSE).
