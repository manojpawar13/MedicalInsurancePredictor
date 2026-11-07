import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["age", "bmi", "children"]
CAT_COLS = ["sex", "smoker", "region"]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
        ]
    )

def make_input_frame(age, sex, bmi, children, smoker, region):
    return pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])
