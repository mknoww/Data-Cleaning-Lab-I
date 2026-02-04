import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def college_train_test(
    csv_path: str = "cc_institution_details.csv",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    College Completion pipeline (Step 4):
    Returns train/test feature matrices + labels after completing all prep steps.
    Target: high_grad_150 (1 if grad_150_value is above the dataset median, else 0)
    """

    # 1) Load the dataset from your repo
    df = pd.read_csv(csv_path)

    # 2) Make sure the graduation rate column is numeric (bad values become NaN)
    df["grad_150_value"] = pd.to_numeric(df["grad_150_value"], errors="coerce")

    # 3) Drop rows where we don't know the graduation rate (can't create the target)
    df = df.dropna(subset=["grad_150_value"])

    # 4) Create the target variable using a data-driven cutoff (the median)
    cutoff = df["grad_150_value"].median()
    df["high_grad_150"] = (df["grad_150_value"] > cutoff).astype(int)

    # 5) Drop columns that are identifiers/text (not useful for prediction)
    #    and columns that would leak the outcome (or are very close to it).
    #    IMPORTANT: We also drop grad_150_value because we used it to define the label.
    drop_cols = [
        "index", "unitid", "chronname", "site", "nicknames", "similar",
        "grad_150_value",
        "grad_100_value", "retain_value",
        "grad_100_percentile", "retain_percentile", "grad_150_percentile",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 6) Split into features (X) and target (y)
    y = df["high_grad_150"]
    X = df.drop(columns=["high_grad_150"])

    # 7) Identify which columns are numeric vs categorical so we can preprocess correctly
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 8) Build a preprocessing pipeline:
    #    - Numeric: fill missing values with median, then standardize
    #    - Categorical: fill missing values with most frequent value, then one-hot encode
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    # 9) Create train/test split (stratify keeps class balance similar in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # 10) Fit preprocessing on TRAIN only, then apply to test
    #     (This avoids leaking info from the test set into scaling/encoding.)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return X_train_prep, X_test_prep, y_train.to_numpy(), y_test.to_numpy()


def job_train_test(
    csv_path: str = "job_placement.csv",
    salary_col: str = "salary",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Job Placement pipeline (Step 4):
    Returns train/test feature matrices + labels after completing all prep steps.
    Target: above_median_salary (1 if salary is above the dataset median, else 0)

    Note: Many placement datasets only include salary for placed students.
    In that common case, this pipeline is modeling "high salary among those with known salary."
    """

    # 1) Load the dataset from your repo
    df = pd.read_csv(csv_path)

    # 2) Confirm the salary column exists and coerce it to numeric
    if salary_col not in df.columns:
        raise ValueError(f"Couldn't find '{salary_col}' in CSV columns. Update salary_col to match your file.")

    df[salary_col] = pd.to_numeric(df[salary_col], errors="coerce")

    # 3) Keep only rows where salary is known (otherwise we can't label above/below median)
    df = df.dropna(subset=[salary_col])

    # 4) Create the target variable using the median salary cutoff
    cutoff = df[salary_col].median()
    df["above_median_salary"] = (df[salary_col] > cutoff).astype(int)

    # 5) Drop salary itself (it directly defines the label -> leakage)
    #    Also drop placement status columns if they exist (often post-outcome information).
    drop_cols = [salary_col]
    for c in df.columns:
        if c.lower() in ["status", "placed"]:
            drop_cols.append(c)

    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # 6) Split into features (X) and target (y)
    y = df["above_median_salary"]
    X = df.drop(columns=["above_median_salary"])

    # 7) Identify numeric vs categorical columns
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 8) Preprocess:
    #    - Numeric: impute median, then standardize
    #    - Categorical: impute most frequent, then one-hot encode
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    # 9) Train/test split with stratification so class balance stays similar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # 10) Fit preprocessing on TRAIN only, then apply to test
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return X_train_prep, X_test_prep, y_train.to_numpy(), y_test.to_numpy()

