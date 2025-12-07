# Transfer Learning Image Classifier

## One-line objective
Build and evaluate an image classifier for a custom dataset using transfer learning (ResNet50) and Grad-CAM interpretability.

## Repo structure
- data/: dataset split into train/val/test (not checked in)
- notebooks/: Jupyter notebooks (train_and_eval.ipynb)
- scripts/: helper scripts (data_utils.py, model_build.py, train.py, gradcam.py)
- models/: trained model weights
- visualizations/: plots, confusion matrix, Grad-CAM images
- requirements.txt
- README.md

## Quick start
1. Create virtual env: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Place dataset in `data/` with train/val/test folders
4. Open `notebooks/train_and_eval.ipynb` and run cells

## Contact
Your Name — your.email@example.com
