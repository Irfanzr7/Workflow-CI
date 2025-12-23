import os
import glob
import json
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
try:
    import dagshub
except Exception:
    dagshub = None
import argparse
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import randint, uniform

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------
# DAGSHUB + MLFLOW INIT
# -------------------------------------------------------------------
def try_init_dagshub(repo_owner="Irfanzr7", repo_name="MSML_IRFAN-ZIYADI-RIZKILLAH"):
    if dagshub is None:
        logging.debug("dagshub not installed; skipping DagsHub init")
        return False
    if not os.getenv("DAGSHUB_USER_TOKEN"):
        logging.warning("DAGSHUB_USER_TOKEN not found; skipping DagsHub init")
        return False
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        logging.info("Initialized DagsHub for %s/%s", repo_owner, repo_name)
        return True
    except Exception as e:
        logging.warning("Failed to init DagsHub: %s", e)
        return False

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def find_processed_csv():
    candidates = [
        os.path.join("weather_preprocessing", "seattle_weather_processed.csv"),
        "seattle_weather_processed.csv",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    matches = glob.glob("**/*processed*.csv", recursive=True)
    if matches:
        logging.info("Found processed csv: %s", matches[0])
        return matches[0]

    raise FileNotFoundError("Dataset hasil preprocessing tidak ditemukan")


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

# -------------------------------------------------------------------
# MAIN TRAINING
# -------------------------------------------------------------------
def main(n_iter=30, test_size=0.2, random_state=42, data_path=None, out_dir="artifacts"):
    # non-fatal DagsHub init
    try_init_dagshub()

    # allow overriding the dataset path via CLI
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        logging.info("Using dataset: %s", data_path)
    else:
        data_path = find_processed_csv()

    df = pd.read_csv(data_path)

    if "weather" not in df.columns:
        raise ValueError("Kolom target 'weather' tidak ditemukan")

    X = df.drop(columns=["weather"])
    y = df["weather"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    models = {
        "logreg": (
            Pipeline([
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=2000))
            ]),
            {
                "model__C": uniform(0.01, 10),
                "model__solver": ["lbfgs", "saga"],
                "model__penalty": ["l2"]
            }
        ),
        "rf": (
            Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestClassifier(random_state=random_state))
            ]),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(3, 30),
                "model__min_samples_split": randint(2, 10)
            }
        )
    }

    best_model = None
    best_f1 = -1

    with mlflow.start_run(run_name="model_tuning"):
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("test_size", test_size)

        for name, (pipe, params) in models.items():
            logging.info("Tuning model: %s", name)

            search = RandomizedSearchCV(
                pipe,
                params,
                n_iter=n_iter,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # ------------------ LOG METRICS ------------------
            mlflow.log_metric(f"{name}_accuracy", acc)
            mlflow.log_metric(f"{name}_f1_weighted", f1)

            # ------------------ LOG PARAMS ------------------
            try:
                mlflow.log_dict(search.best_params_, f"{name}_best_params.json")
            except Exception:
                mlflow.log_param(f"{name}_best_params", json.dumps(search.best_params_))

            # ------------------ LOG ARTIFACTS ------------------
            os.makedirs(out_dir, exist_ok=True)

            report_path = os.path.join(out_dir, f"classification_report_{name}.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(report_path)

            model_path = os.path.join(out_dir, f"model_{name}.joblib")
            joblib.dump(search.best_estimator_, model_path)
            mlflow.log_artifact(model_path)

            if f1 > best_f1:
                best_f1 = f1
                best_model = (name, search.best_estimator_)

        # ------------------ LOG BEST MODEL ------------------
        if best_model:
            name, model = best_model
            mlflow.log_param("best_model", name)
            mlflow.sklearn.log_model(model, artifact_path="best_model")

            # Save the MLflow model locally for downstream Docker build step
            os.makedirs(out_dir, exist_ok=True)
            model_export_path = os.path.join(out_dir, "model")
            mlflow.sklearn.save_model(model, model_export_path)

    if best_model:
        print("Training & tuning selesai. Best model:", best_model[0])
    else:
        print("Training & tuning selesai. No best model selected.")

# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune models for Seattle weather dataset")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of iterations for RandomizedSearchCV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, default=None, help="Path to processed dataset CSV")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="Directory to save artifacts/models")

    args = parser.parse_args()

    try:
        main(n_iter=args.n_iter, test_size=args.test_size, random_state=args.random_state,
             data_path=args.data_path, out_dir=args.out_dir)
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(2)
    except Exception as e:
        logging.exception("Unhandled error during training: %s", e)
        sys.exit(1)
