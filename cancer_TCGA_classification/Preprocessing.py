import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer

# =============================================================================
def split_dataset(
    dataset,
    label_col="Cancer_Type",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """
    Stratified Train/Validation/Test split.
    """

    train_df, temp_df = train_test_split(
        dataset,
        test_size=test_size + val_size,
        stratify=dataset[label_col],
        random_state=random_state,
    )

    relative_val = val_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df[label_col],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# =============================================================================
def preprocess(dataset):

    print("=" * 60)
    print("Splitting Dataset")
    print("=" * 60)

    train_df, val_df, test_df = split_dataset(dataset)

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)


    

    print()

    print("=" * 60)
    print("Learning Imputation Statistics")
    print("=" * 60)

    # Numerical columns (gene expression)
    numerical_columns_selector = selector(dtype_include=["number"])

    # Categorical columns (Sample_Type, Cancer_Type)
    categorical_columns_selector = selector(dtype_exclude=["number"])

    numerical_columns = numerical_columns_selector(train_df)
    categorical_columns = categorical_columns_selector(train_df)

    print("Number of numerical columns:", len(numerical_columns))
    print("Categorical columns:", categorical_columns)
    #print("numerical columns:", numerical_columns)

    # Numerical (genes)
    gene_imputer = SimpleImputer(strategy="mean")

    train_df[numerical_columns] = gene_imputer.fit_transform(
    train_df[numerical_columns])

    val_df[numerical_columns] = gene_imputer.transform(
    val_df[numerical_columns])

    test_df[numerical_columns] = gene_imputer.transform(
    test_df[numerical_columns])

    os.makedirs("./models", exist_ok=True)
    joblib.dump(gene_imputer, "./models/gene_imputer.pkl")


    print()

    print("=" * 60)
    print("Missing Values After Imputation")
    print("=" * 60)

    print("Train:", train_df.isnull().sum().sum())
    print("Validation:", val_df.isnull().sum().sum())
    print("Test:", test_df.isnull().sum().sum())

    return train_df, val_df, test_df

# =============================================================================
def save_processed_data(
    train_df,
    val_df,
    test_df,
    output_dir="./preprocessed_data",
):

    

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(
        os.path.join(output_dir, "train.csv"),
        index=False,
    )

    val_df.to_csv(
        os.path.join(output_dir, "validation.csv"),
        index=False,
    )

    test_df.to_csv(
        os.path.join(output_dir, "test.csv"),
        index=False,
    )

    print()

    print("=" * 60)
    print("Datasets Saved")
    print("=" * 60)

    print(output_dir)

# =============================================================================
#MAIN 
dataset = pd.read_csv("TCGA_Cancer_Dataset.csv")
train_df, val_df, test_df = preprocess(dataset)
save_processed_data(train_df, val_df, test_df)
