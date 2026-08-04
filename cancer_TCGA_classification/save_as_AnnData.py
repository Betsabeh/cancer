import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import LabelEncoder

import scanpy as sc
import anndata as ad

import joblib
#=========================================================
#==========================================================
def read_expression(Data):
    expression = Data.drop(
        columns=["Cancer_Type", "tissue_type"],
        errors="ignore"
    )

    print("=" * 60)
    print("Gene expression data size:")
    print(expression.shape)

    return expression
#=========================================================
def read_labels(Data,label_column_name, encoder=None):
    Labels = Data[[label_column_name]].copy()

    if encoder is None:
        encoder = LabelEncoder()
        print(np.unique(Labels[label_column_name]))
        Labels["label_id"] = encoder.fit_transform(Labels[label_column_name])
    else:
        print(np.unique(Labels[label_column_name]))
        Labels["label_id"] = encoder.transform(Labels[label_column_name])

    print(np.unique(Labels['label_id']))

    return Labels, encoder
#=========================================================
def create_anndata(expression_df,
                   labels,
                   label_column):
 
    # --------------------------
    # Create AnnData
    # --------------------------
    adata = ad.AnnData(X=expression_df.values)
    # --------------------------
    # Sample information
    # --------------------------

    adata.obs.index = expression_df.index

    adata.obs["patient_id"] = expression_df.index

    adata.obs[label_column] = labels[label_column].values

    adata.obs["label_id"] = labels["label_id"].values

    # --------------------------
    # Gene information
    # --------------------------
    adata.var.index = expression_df.columns
    adata.var["gene_name"] = expression_df.columns

    print("=" * 50)
    print("AnnData")
    print("=" * 50)
    print(adata)

    return adata
#=========================================================
def prepare_h5ad(df, label_column, encoder=None):
    expression = read_expression(df)
    labels, encoder = read_labels(df, label_column, encoder)
    adata = create_anndata(expression, labels, label_column)
    return adata, encoder
#=========================================================
def save_h5ad(adata,
              output_path):

    adata.write(output_path)

    print("=" * 50)
    print("Saved Successfully")
    print(output_path)
    print("=" * 50)

#-------------------------------READ DATA------------------
Train_data =pd.read_csv('F:\\ML\\preprocessed_data\\train.csv',index_col=0)
#print("-----------person 1:10----------------")
print(Train_data.iloc[0:10])

Test_data =pd.read_csv('F:\\ML\\preprocessed_data\\test.csv', index_col=0)
#print("-----------person 1:10----------------")
#print(Test_data.iloc[0:10])

valid_data =pd.read_csv('F:\\ML\\preprocessed_data\\validation.csv', index_col=0)
#print("-----------person 1:10----------------")
#print(valid_data.iloc[0:10])

print("-----------prepare Train_h5ad-----------------------")
print(np.shape(Train_data))
train_adata, encoder = prepare_h5ad(Train_data, "Cancer_Type")

print("-----------prepare valid_h5ad----------------")
print(np.shape(valid_data))
valid_adata, _ = prepare_h5ad(valid_data, "Cancer_Type",encoder)
print("-----------prepare test_h5ad----------------")
print(np.shape(Test_data))
test_adata, _ = prepare_h5ad(Test_data, "Cancer_Type",encoder)

save_h5ad(train_adata, r"F:\ML\preprocessed_data\train_scgpt.h5ad")
save_h5ad(valid_adata, r"F:\ML\preprocessed_data\valid_scgpt.h5ad")
save_h5ad(test_adata, r"F:\ML\preprocessed_data\test_scgpt.h5ad")

# save encoder
joblib.dump(encoder, "label_encoder.pkl")
