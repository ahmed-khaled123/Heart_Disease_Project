
# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

This repo implements the full pipeline described in your spec: preprocessing, PCA, feature selection, supervised & unsupervised learning, hyperparameter tuning, model export, Streamlit UI, and Ngrok deployment notes.

> **Dataset:** Place the UCI Heart Disease CSV at `data/heart_disease.csv` (recommended Cleveland subset).  
> A small synthetic file is provided at `data/sample_heart_disease.csv` for smoke testing only.

## Quick Start

```bash
# 1) Create & activate env (example)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Put real dataset CSV at:
#    Heart_Disease_Project/data/heart_disease.csv

# 4) Train & evaluate
python -m src.train_eval

# 5) Unsupervised analysis
python -m src.unsupervised

# 6) Run Streamlit UI
streamlit run ui/app.py
```

Artifacts will be saved to `models/` and `results/`. If `data/heart_disease.csv` is missing, the training code will use the synthetic sample for a quick test.

## Notebooks

Notebooks in `notebooks/` mirror each step:

- `01_data_preprocessing.ipynb`
- `02_pca_analysis.ipynb`
- `03_feature_selection.ipynb`
- `04_supervised_learning.ipynb`
- `05_unsupervised_learning.ipynb`
- `06_hyperparameter_tuning.ipynb`

They rely on `src/utils.py` and read from `data/heart_disease.csv` (or the synthetic sample).

## Ngrok Deployment (Bonus)

Steps in `deployment/ngrok_setup.txt` show how to expose the local Streamlit app.

## File Structure

Heart_Disease_Project/
├── data/
│   ├── heart_disease.csv                 # <- put the real dataset here
│   └── sample_heart_disease.csv          # synthetic sample (for smoke tests)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── models/
│   └── final_model.pkl                   # created after training
├── src/
│   ├── utils.py
│   ├── train_eval.py
│   └── unsupervised.py
├── ui/
│   └── app.py
├── results/
│   ├── evaluation_metrics.json
│   ├── evaluation_metrics.txt
│   ├── roc_curves.png
│   ├── kmeans_elbow.png
│   └── hierarchical_dendrogram.png
├── deployment/
│   └── ngrok_setup.txt
├── requirements.txt
├── README.md
└── .gitignore

## Notes

- Pipelines include preprocessing (impute + scale + one-hot), optional PCA (95% variance), and the classifier.
- Feature selection techniques (RFE, Chi-Square, Feature Importance) are demonstrated in the notebooks.
- Hyperparameter tuning is included for RandomForest (RandomizedSearchCV) and SVM (GridSearchCV).
- The final model is selected based on the tuned models' F1-score.
