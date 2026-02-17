import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Load data
train_df = pd.read_csv("./Train.csv")
test_df  = pd.read_csv("./Test.csv")

# Separate features & label
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"]

X_test = test_df.drop(columns=["Label"], errors="ignore")

# Identify columns
id_column = ["ID"]
numeric_features = X_train.columns.drop(id_column)

# Preprocessing pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("drop_id", "drop", id_column)
    ]
)

# Fit on TRAIN, transform BOTH
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# Convert to DataFrame
feature_names = numeric_features  # ID dropped

train1 = pd.DataFrame(X_train_processed, columns=feature_names)
train1["Label"] = y_train.values

test1 = pd.DataFrame(X_test_processed, columns=feature_names)

# Save processed datasets
train1.to_csv("Train1.csv", index=False)
test1.to_csv("Test1.csv", index=False)


print("Saved: Train1.csv and Test1.csv")
