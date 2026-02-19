import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# ---------------------------------------------------------
# Data Loading and Saving
# ---------------------------------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from an Excel file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_excel(file_path)
    return df

def save_excel(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves DataFrame to Excel.
    """
    print(f"Saving file to: {file_path}")
    df.to_excel(file_path, index=False)

def get_output_paths(input_path: str) -> Tuple[str, str]:
    """
    Generates output file paths based on the input filename.
    Returns: (folded_data_path, analysis_path)
    """
    base_dir = os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    name_no_ext, ext = os.path.splitext(file_name)
    
    folded_path = os.path.join(base_dir, f"{name_no_ext}_folded{ext}")
    analysis_path = os.path.join(base_dir, "stratification_analysis.xlsx")
    
    return folded_path, analysis_path

# ---------------------------------------------------------
# Core Stratification Logic
# ---------------------------------------------------------
def get_patient_compositions(df: pd.DataFrame, patient_col: str, y_col: str) -> List[Dict]:
    """
    Aggregates the dataframe by patient to determine the 'content' of each patient.
    
    Returns a list of dictionaries, where each dictionary represents a patient:
    {
        'id': patient_id,
        'total_samples': int,
        'class_counts': {class_label: count},
        'indices': [list of original dataframe indices for this patient]
    }
    """
    patients = []
    # Identify unique classes in the entire dataset
    unique_classes = sorted(df[y_col].unique())
    
    # Group by patient
    grouped = df.groupby(patient_col)
    
    for pid, group in grouped:
        counts = group[y_col].value_counts().to_dict()
        # Ensure all classes are present in the dictionary (even if count is 0)
        class_counts = {k: counts.get(k, 0) for k in unique_classes}
        
        patients.append({
            'id': pid,
            'total_samples': len(group),
            'class_counts': class_counts,
            'indices': group.index.tolist()
        })
        
    return patients

def assign_folds_greedy(
    patients: List[Dict], 
    unique_classes: List, 
    n_splits: int, 
    random_state: int
) -> Dict[str, int]:
    """
    Assigns folds to patients using a greedy approach to balance class distributions.
    
    Strategy:
    1. Sort patients by total samples (descending) to handle largest chunks first (better bin packing).
    2. For each patient, calculate the CHANGE in error (delta) for each fold.
    3. Determine global improvement: choosing the fold where adding the patient 
       reduces the squared error from the target the most.
    """
    rng = np.random.RandomState(random_state)
    
    # Shuffle first for randomness, then sort by size descending for packing efficiency
    rng.shuffle(patients)
    patients.sort(key=lambda x: x['total_samples'], reverse=True)
    
    # Calculate global targets (Total samples of class C / n_splits)
    global_counts = {c: 0 for c in unique_classes}
    for p in patients:
        for c in unique_classes:
            global_counts[c] += p['class_counts'][c]
            
    target_counts_per_fold = {c: global_counts[c] / n_splits for c in unique_classes}
    
    # Initialize fold stat trackers
    # fold_counts[k][c] = number of samples of class c in fold k
    fold_counts = {k: {c: 0 for c in unique_classes} for k in range(1, n_splits + 1)}
    
    # Result dictionary: patient_id -> fold_number
    patient_fold_map = {}
    
    for p in patients:
        id_ = p['id']
        p_counts = p['class_counts']
        
        best_fold = -1
        # We look for the lowest (most negative) delta. 
        # Ideally, adding a patient reduces error, so delta is negative.
        best_delta = float('inf')
        
        # Try finding the best fold for this patient
        for k in range(1, n_splits + 1):
            current_fold_counts = fold_counts[k]
            
            current_error = 0
            new_error = 0
            
            for c in unique_classes:
                target = target_counts_per_fold[c]
                count = current_fold_counts[c]
                added = p_counts[c]
                
                # Error before adding patient
                current_error += (count - target) ** 2
                
                # Error after adding patient
                new_error += (count + added - target) ** 2
            
            # The improvement (or worsening) in score
            delta = new_error - current_error
            
            if delta < best_delta:
                best_delta = delta
                best_fold = k
        
        # Assign patient to the best fold found
        patient_fold_map[id_] = best_fold
        
        # Update the tracker
        for c in unique_classes:
            fold_counts[best_fold][c] += p_counts[c]
            
    return patient_fold_map

# ---------------------------------------------------------
# Analysis and Reporting
# ---------------------------------------------------------
def create_analysis_report(df: pd.DataFrame, fold_col: str, y_col: str) -> pd.DataFrame:
    """
    Creates a summary DataFrame showing the distribution of classes and total samples per fold.
    """
    # Group by Fold and Label
    report = df.groupby([fold_col, y_col]).size().unstack(fill_value=0)
    
    # Add Total Samples column
    report['Total Samples'] = report.sum(axis=1)
    
    # Calculate percentages for class distribution
    class_cols = [c for c in report.columns if c != 'Total Samples']
    for c in class_cols:
        report[f'{c} (%)'] = (report[c] / report['Total Samples'] * 100).round(2)
        
    return report

# ---------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------
def run_stratification(
    input_path: str,
    n_splits: int,
    patient_col: str,
    sample_col: str,
    y_col: str,
    random_state: int
):
    print("--- Starting Custom Patient-Level Stratification ---")
    
    # 1. Load Data
    df = load_data(input_path)
    
    # Basic Validation
    required_cols = [patient_col, sample_col, y_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the input Excel file.")
            
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Unique Label Classes: {sorted(df[y_col].unique())}")
    
    # 2. Prepare Patient Objects
    patients = get_patient_compositions(df, patient_col, y_col)
    unique_classes = sorted(df[y_col].unique())
    
    print(f"Processing {len(patients)} unique patients...")
    
    # 3. Algorithm: Assign Folds
    patient_fold_map = assign_folds_greedy(patients, unique_classes, n_splits, random_state)
    
    # 4. Map results back to DataFrame
    # Create the FOLD column based on patient ID
    df['FOLD'] = df[patient_col].map(patient_fold_map)
    
    # 5. Generate Analysis
    analysis_df = create_analysis_report(df, 'FOLD', y_col)
    
    # 6. Save Outputs
    folded_path, analysis_path = get_output_paths(input_path)
    
    save_excel(df, folded_path)
    save_excel(analysis_df, analysis_path) # Saving analysis to a separate file as requested, or could be a sheet
    
    print("\n--- Stratification Analysis ---")
    print(analysis_df)
    print("\nProcessing complete.")

# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Custom Patient-Aware Stratified K-Fold Cross-Validation generator."
    )
    
    parser.add_argument("--input_path", type=str, default="./path_to_your_data.xlsx", 
                        help="Path to the input .xlsx file.")
    parser.add_argument("--n_splits", type=int, default=5, 
                        help="Number of folds for cross-validation.")
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Seed for random permutation.")
    parser.add_argument("--patient_col", type=str, default="ID", 
                        help="Column name containing Patient IDs.")
    parser.add_argument("--sample_col", type=str, default="SAMPLES", 
                        help="Column name containing Sample IDs.")
    parser.add_argument("--y_col", type=str, default="ACTIVITY", 
                        help="Column name containing the target labels/classes.")

    args = parser.parse_args()

    try:
        run_stratification(
            input_path=args.input_path,
            n_splits=args.n_splits,
            patient_col=args.patient_col,
            sample_col=args.sample_col,
            y_col=args.y_col,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)