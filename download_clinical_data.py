import requests
import json
import pandas as pd
import os

def download_gdc_clinical_data():
    # Base URL for GDC API
    base_url = "https://api.gdc.cancer.gov/"
    
    # List of common cancer types in GDC (33 cancer types)
    cancer_types = [
        "TCGA-BRCA", "TCGA-GBM", "TCGA-OV", "TCGA-LUAD", "TCGA-UCEC", 
        "TCGA-KIRC", "TCGA-HNSC", "TCGA-LGG", "TCGA-THCA", "TCGA-LUSC",
        "TCGA-PRAD", "TCGA-SKCM", "TCGA-COAD", "TCGA-STAD", "TCGA-BLCA",
        "TCGA-LIHC", "TCGA-CESC", "TCGA-KIRP", "TCGA-SARC", "TCGA-LAML",
        "TCGA-ESCA", "TCGA-PAAD", "TCGA-PCPG", "TCGA-READ", "TCGA-TGCT",
        "TCGA-THYM", "TCGA-KICH", "TCGA-ACC", "TCGA-MESO", "TCGA-UVM",
        "TCGA-DLBC", "TCGA-UCS", "TCGA-CHOL"
    ]
    
    # Create directory to store data
    if not os.path.exists("clinical_data"):
        os.makedirs("clinical_data")
    
    for cancer in cancer_types:
        print(f"Downloading clinical data for {cancer}...")
        
        # Construct the query
        query = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": [cancer]
                        }
                    }
                ]
            },
            "format": "JSON",
            "fields": "cases.case_id,cases.demographic,cases.diagnoses,cases.exposures",
            "size": 1000
        }
        
        # Make the API request
        response = requests.post(f"{base_url}cases", json=query)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract clinical data
            clinical_data = []
            for case in data.get('data', {}).get('hits', []):
                case_data = {
                    'case_id': case.get('case_id'),
                    'demographic': case.get('demographic', {}),
                    'diagnoses': case.get('diagnoses', []),
                    'exposures': case.get('exposures', [])
                }
                clinical_data.append(case_data)
            
            # Save to file
            with open(f"clinical_data/{cancer}_clinical.json", 'w') as f:
                json.dump(clinical_data, f, indent=2)
            
            # Convert to DataFrame and save as CSV
            df = pd.json_normalize(clinical_data)
            df.to_csv(f"clinical_data/{cancer}_clinical.csv", index=False)
            
            print(f"Successfully downloaded data for {cancer}")
        else:
            print(f"Failed to download data for {cancer}. Status code: {response.status_code}")

if __name__ == "__main__":
    download_gdc_clinical_data()
