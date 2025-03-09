import requests
import pandas as pd
import json

# Base URL for GDC API
base_url = "https://api.gdc.cancer.gov/"

# Get list of all projects (cancer types)
projects_endpoint = base_url + "projects"
response = requests.get(projects_endpoint, params={"size": "1000"})
projects = response.json()["data"]["hits"]

# Filter for TCGA projects (33 cancer types)
tcga_projects = []
for project in projects:
    # Check if project name contains "TCGA"
    if "TCGA" in project.get("project_id", ""):
        tcga_projects.append(project)

# Initialize empty list to store all clinical data
all_clinical_data = []

# Loop through each cancer type
for project in tcga_projects:
    project_id = project["project_id"]
    print(f"Downloading clinical data for {project_id}...")
    
    # Get cases for this project
    cases_endpoint = base_url + "cases"
    cases_params = {
        "size": "1000",
        "project.project_id": project_id,
        "expand": "demographic,diagnoses,exposures"
    }
    
    response = requests.get(cases_endpoint, params=cases_params)
    cases = response.json()["data"]["hits"]
    
    # Extract clinical data for each case
    for case in cases:
        clinical_data = {
            "project_id": project_id,
            "case_id": case["case_id"],
            "submitter_id": case["submitter_id"],
            "primary_site": case.get("primary_site", ""),
            "disease_type": case.get("disease_type", ""),
            "gender": case.get("demographic", {}).get("gender", ""),
            "age_at_diagnosis": case.get("demographic", {}).get("age_at_index", ""),
            "vital_status": case.get("demographic", {}).get("vital_status", ""),
            "days_to_death": case.get("demographic", {}).get("days_to_death", ""),
            "race": case.get("demographic", {}).get("race", ""),
            "ethnicity": case.get("demographic", {}).get("ethnicity", "")
        }
        
        # Get diagnosis information if available
        if "diagnoses" in case and len(case["diagnoses"]) > 0:
            diagnosis = case["diagnoses"][0]
            clinical_data.update({
                "tumor_stage": diagnosis.get("tumor_stage", ""),
                "tumor_grade": diagnosis.get("tumor_grade", ""),
                "prior_treatment": diagnosis.get("prior_treatment", ""),
                "tissue_or_organ_of_origin": diagnosis.get("tissue_or_organ_of_origin", "")
            })
            
        all_clinical_data.append(clinical_data)

# Convert to DataFrame and save to CSV
df = pd.DataFrame(all_clinical_data)
output_file = "tcga_clinical_data.csv"
df.to_csv(output_file, index=False)
print(f"\nDownload complete! Data saved to {output_file}")
print(f"Total cases downloaded: {len(df)}")


