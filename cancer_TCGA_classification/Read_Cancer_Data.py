import gzip
import pandas as pd
import numpy as np

# =============================================================================
# Step 1. Load Gene Expression Data
# =============================================================================

Gene_expression_Dataset_name = r"F:\research\research\Multimodal_Disease_Prediction\data\EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"

with gzip.open(Gene_expression_Dataset_name, 'rt') as f:
    gene_expr = pd.read_csv(f, sep="\t", index_col=0)

print("="*60)
print("Gene Expression Dataset")
print("="*60)
print("Shape:", gene_expr.shape)
print("Rows    : Genes")
print("Columns : TCGA Samples of Patient")
print()

# =============================================================================
# Step 2. Load Clinical Data
# =============================================================================

Clinical_Dataset_name = r"F:\research\research\Multimodal_Disease_Prediction\data\TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

with gzip.open(Clinical_Dataset_name, 'rt') as f:
    clinical = pd.read_csv(f, sep="\t", index_col=0)

print("="*60)
print("Clinical Dataset")
print("="*60)
print("Shape:", clinical.shape)
print()

print("="*60)
print("An example of gene expression:\n", gene_expr.head())
print("An example of related clinical data:\n", clinical.head())
# =============================================================================
# Step 3. Match Samples
# =============================================================================

common_samples = gene_expr.columns.intersection(clinical.index)
#print(common_samples)

print("="*60)
print("Sample Matching")
print("="*60)
print("Expression Samples :", len(gene_expr.columns))
print("Clinical Samples   :", len(clinical.index))
print("Matched Samples    :", len(common_samples))
print()

# Keep only common samples

gene_expr = gene_expr[common_samples]
clinical = clinical.loc[common_samples]


# =============================================================================
# Step 4. Transpose Gene Matrix
# =============================================================================

gene_expr = gene_expr.T

print("="*60)
print("Transposed Gene Expression")
print("="*60)
print(gene_expr.shape)
print()

# =============================================================================
# Step 5. Explore Clinical Columns
# =============================================================================

print("="*60)
print("Clinical Columns")
print("="*60)

for c in clinical.columns:
    print(c)

print()

# =============================================================================


TARGET_COLUMN = "_primary_disease"   
TISSUE_COLUMN = "sample_type"
# =============================================================================
# Step 6. Build Final Dataset
# =============================================================================

dataset = gene_expr.copy()

dataset["Cancer_Type"] = clinical[TARGET_COLUMN]
dataset["tissue_type"] = clinical[TISSUE_COLUMN]

print("="*60)
print("Final Dataset")
print("="*60)
print(dataset.shape)
print()

print(dataset.head())

# =============================================================================
# Step 7. Remove Missing Labels
# =============================================================================

dataset = dataset.dropna(subset=["Cancer_Type"])

print("="*60)
print("After Removing Missing Cancer Labels")
print("="*60)
print(dataset.shape)
print()

# =============================================================================
# Step 8. Cancer Distribution
# =============================================================================

print("="*60)
print("Cancer Type Distribution")
print("="*60)

print(dataset["Cancer_Type"].value_counts())

# =============================================================================
# Step 9. Missing Values
# =============================================================================

print("="*60)
print("Missing Values")
print("="*60)

print("Total Missing:", dataset.isnull().sum().sum())
print(dataset.shape)

print(dataset["Cancer_Type"].isnull().sum())

missing_per_patient = dataset.isnull().sum(axis=1)
print(missing_per_patient.describe())

print("Patients with NO missing values:",
      (missing_per_patient == 0).sum())

print("Patients with missing values:",
      (missing_per_patient > 0).sum())
'''
for i in range(len(dataset)):
    if dataset.iloc[i].isnull().sum()>0:
        print(dataset.iloc[i])
'''
# =============================================================================
# Step 10. Save Dataset
# =============================================================================

dataset.to_csv("TCGA_Cancer_Dataset.csv")

print()
print("="*60)
print("Dataset saved as TCGA_Cancer_Dataset.csv")
print("="*60)


