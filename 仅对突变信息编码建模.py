import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


INPUT_FILE = "GFP_data_avGFP.xlsx"

MUTATION_COL = "aaMutations"
LABEL_COL = "Brightness"

PROTEIN_LENGTH = 238
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# 一般先设为 0，除非你确认数据位点需要整体 +1
# POSITION_OFFSET = 0
POSITION_OFFSET = 1


mut_pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def read_input_file(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith((".tsv", ".txt")):
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError("Unsupported file format")


def parse_mutation_string(mut_str):
    if pd.isna(mut_str):
        return []

    mut_str = str(mut_str).strip()

    if mut_str == "" or mut_str.lower() in ["wt", "wildtype", "none", "nan"]:
        return []

    mutations = []

    for item in mut_str.split(":"):
        item = item.strip()
        match = mut_pattern.fullmatch(item)

        if match is None:
            raise ValueError(f"Invalid mutation format: {item}")

        wt_aa, pos_str, mut_aa = match.groups()
        pos = int(pos_str)

        if wt_aa not in AA_TO_IDX:
            raise ValueError(f"Invalid WT amino acid: {wt_aa}")

        if mut_aa not in AA_TO_IDX:
            raise ValueError(f"Invalid mutant amino acid: {mut_aa}")

        corrected_pos = pos + POSITION_OFFSET

        if corrected_pos < 1 or corrected_pos > PROTEIN_LENGTH:
            raise ValueError(
                f"Position out of range: {item}, corrected position = {corrected_pos}"
            )

        mutations.append((wt_aa, corrected_pos, mut_aa))

    return mutations


def build_mutation_onehot(mutation_series):
    """
    输出:
        X: shape = [n_samples, 238 * 20]
        mutation_count: 每个样本的突变数量
    """
    n = len(mutation_series)
    X = np.zeros((n, PROTEIN_LENGTH * len(AA_LIST)), dtype=np.int8)
    mutation_count = np.zeros(n, dtype=np.int16)

    for i, mut_str in enumerate(mutation_series):
        mutations = parse_mutation_string(mut_str)
        mutation_count[i] = len(mutations)

        for wt_aa, pos, mut_aa in mutations:
            pos_idx = pos - 1
            aa_idx = AA_TO_IDX[mut_aa]

            feature_idx = pos_idx * len(AA_LIST) + aa_idx
            X[i, feature_idx] = 1

    feature_names = [
        f"pos{pos}_{aa}"
        for pos in range(1, PROTEIN_LENGTH + 1)
        for aa in AA_LIST
    ]

    X_df = pd.DataFrame(X, columns=feature_names)
    X_df["mutation_count"] = mutation_count

    return X_df


df = read_input_file(INPUT_FILE)

df = df.dropna(subset=[MUTATION_COL, LABEL_COL]).reset_index(drop=True)

# 如果数据很多，可以先抽样；正式建模建议不要随便抽样
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

X = build_mutation_onehot(df[MUTATION_COL])
y = df[LABEL_COL].astype(float)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("R2:", r2)
print("RMSE:", rmse)
print("MAE:", mae)

import joblib

MODEL_FILE = "xgb_gfp_brightness_model.joblib"

joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "protein_length": PROTEIN_LENGTH,
        "aa_list": AA_LIST,
        "aa_to_idx": AA_TO_IDX,
        "position_offset": POSITION_OFFSET,
        "mutation_col": MUTATION_COL,
        "label_col": LABEL_COL,
    },
    MODEL_FILE
)

print(f"Model saved to: {MODEL_FILE}")