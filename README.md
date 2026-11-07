# Medical Insurance Cost Prediction

A regression mini-project with EDA notebooks, reproducible preprocessing and model training, saved model artifacts, and a Streamlit prediction interface.

## Project structure
- `Insurance.csv`: source dataset
- `notebooks/01_EDA.ipynb`: original EDA notebook, preserved
- `notebooks/02_Data_Preprocessing.ipynb`: duplicate handling, split, scaling, encoding
- `notebooks/03_Model_Training_Evaluation.ipynb`: Gradient Boosting training and evaluation
- `models/`: fitted model and fitted preprocessor
- `src/`: reusable input and prediction functions
- `app.py`: Streamlit application

## Setup (macOS, VS Code terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Reproduce the model
Open `notebooks/02_Data_Preprocessing.ipynb` and `notebooks/03_Model_Training_Evaluation.ipynb` in order and run all cells. The training notebook writes the `.pkl` files into `models/`.

## Modelling notes
- Exact duplicate rows are removed (one duplicate in the uploaded CSV).
- Train/test split: 80/20, `random_state=42`.
- Numerical features: StandardScaler.
- Categorical features: OneHotEncoder with `drop='first'` and `handle_unknown='ignore'`.
- Selected estimator: `GradientBoostingRegressor(random_state=42)`.
- The target (`charges`) is modelled on its original scale.
- Preprocessing is fitted only on the training split to avoid data leakage.

The prediction is an educational model estimate, not an insurance quote.
