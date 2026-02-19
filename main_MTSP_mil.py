import os
import argparse
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score

# Imports from your local modules
from utils.misc import set_seeds, load_dataframe, compute_metrics_from_cm, plot_confmx
from utils.trainer import train_model, validate_model
from aggregator.MTSP_MIL import MTSPMIL, AB1_OnlyAttn, AB2_AttnTransformer, AB3_AttnTransProto, AB4_AttnTransPyramid


# --- CONFIGURATION & REPRODUCIBILITY ---
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seeds(42, use_cuda=(device == "cuda"))


# --- LOSSES ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --- HELPERS ---
def _to_label_array(y):
    """
    Ensure predictions/targets are 1D integer label arrays.
    If y are probabilities/logits (2D), converts via argmax.
    """
    y = np.asarray(y)
    if y.ndim > 1:
        return y.argmax(axis=1).astype(int)
    return y.astype(int)


def _ensure_probs(arr, n_classes):
    """Ensures the output is a probability distribution (softmax)."""
    a = np.asarray(arr)
    if a.ndim == 1:
        oh = np.zeros((a.shape[0], n_classes), dtype=float)
        oh[np.arange(a.shape[0]), a.astype(int)] = 1.0
        return oh
    if a.ndim == 2 and a.shape[1] == n_classes:
        s = a.sum(axis=1)
        if not np.allclose(s, 1.0):
            exp = np.exp(a - a.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        return a
    raise ValueError("Predictions shape incompatible with n_classes.")


def _standardize_confmat(y_true, y_pred, class_labels):
    """
    Build a confusion matrix with a fixed label set so that the shape is always len(class_labels) x len(class_labels).
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)
    return confusion_matrix(y_true, y_pred, labels=class_labels)


def make_mil_model(name: str, n_classes: int, L: int):
    name = name.strip()
    if name == "MTSPMIL":
        return MTSPMIL(n_classes=n_classes, L=L).to(device)
    if name == "AB1_OnlyAttn":
        return AB1_OnlyAttn(n_classes=n_classes, L=L).to(device)
    if name == "AB2_AttnTransformer":
        return AB2_AttnTransformer(n_classes=n_classes, L=L).to(device)
    if name == "AB3_AttnTransProto":
        return AB3_AttnTransProto(n_classes=n_classes, L=L).to(device)
    if name == "AB4_AttnTransPyramid":
        return AB4_AttnTransPyramid(n_classes=n_classes, L=L).to(device)
    raise ValueError(f"Unknown MIL model: {name}")

# --- DATA LOADING AND VALIDATION ---

def _prepare_label_series(df: pd.DataFrame, label_col: str, classes: int):
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    if classes == 2:
        df[label_col] = df[label_col].replace({2: 1})
    return df


def find_embedding_path(sample_id: str, directories: List[str], embedding_model: str) -> Optional[str]:
    """
    Searches for the embedding file strictly in:
      directory/embedding_model/sample_id.npy
    """
    filename = f"{sample_id}.npy"
    
    for directory in directories:
        # STRICT structure: base_dir / embedding_model / sample_id.npy
        full_path = os.path.join(directory, embedding_model, filename)
        if os.path.exists(full_path):
            return full_path
            
    return None


def load_and_validate_embeddings(df: pd.DataFrame, embedding_dirs: List[str], label_col: str, 
                                 folds_col: str, classes: int, embedding_model: str):
    """
    1. Validates that ALL samples in the DF exist in the specific model subfolder.
    2. Loads the embeddings into the DataFrame.
    """
    print(f"\n[Data] Loading data from base directories: {embedding_dirs}")
    print(f"[Data] Enforcing structure: BASE_DIR/{embedding_model}/SAMPLE_ID.npy")
    
    # 1. Clean and Prepare Labels
    req_cols = ["SAMPLES", label_col, folds_col]
    if not all(col in df.columns for col in req_cols):
        raise ValueError(f"Missing required columns in Excel: {req_cols}")
        
    df = _prepare_label_series(df, label_col, classes)
    
    # 2. Validation & Loading Loop
    embeddings_list = []
    missing_samples = []
    
    # Iterate to maintain order
    for idx, row in df.iterrows():
        sid = row["SAMPLES"] + "_" + embedding_model + "_embeddings" 
        path = find_embedding_path(sid, embedding_dirs, embedding_model)
        
        if path is None:
            missing_samples.append(sid)
        else:
            try:
                # Load tensor (npy)
                emb = np.load(path)
                # Handle different dimensions if necessary (e.g., [1, N, D] -> [N, D])
                if isinstance(emb, np.ndarray):
                    emb = emb.squeeze(1) if emb.ndim == 3 else emb
                # Convert to Tensor here if preferred, or later in loop
                embeddings_list.append(torch.tensor(emb, dtype=torch.float32))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                missing_samples.append(sid)

    # 3. Check for errors
    if len(missing_samples) > 0:
        print("\n" + "!"*50)
        print(f"ERROR: Missing embeddings for {len(missing_samples)} samples.")
        print(f"Expected location example: .../{embedding_model}/{missing_samples[0]}.npy")
        print(f"First 5 missing: {missing_samples[:5]}")
        print("!"*50 + "\n")
        sys.exit(1) # Stop execution if data is missing
        
    df["embeddings"] = embeddings_list
    print(f"[Data] Successfully loaded {len(df)} embeddings.")
    return df


# --- CORE PIPELINE ---

def run_cross_validation(df: pd.DataFrame,
                         label_col: str,
                         folds_col: str,
                         mil_model_name: str,
                         lr: float,
                         epochs: int,
                         use_class_weights: bool,
                         loss_name: str,
                         results_root: str,
                         run_tag: str):
    """
    Performs Cross-Validation using the 'FOLD' column in the DataFrame.
    """
    # Determine Feature Dimension (L)
    L = df["embeddings"].iloc[0].shape[-1]
    
    class_labels = np.array(sorted(np.unique(df[label_col].astype(int))))
    n_classes = len(class_labels)
    folds = sorted(df[folds_col].unique())
    
    print(f"\n[Setup] Classes: {class_labels} (n={n_classes}), Feature Dim (L): {L}")
    print(f"[Setup] Folds found: {folds}")

    # Output directories
    base_dir = os.path.join(results_root, run_tag, "cv")
    scores_dir = os.path.join(base_dir, "output_scores")
    plots_dir = os.path.join(base_dir, "plots") 
    fold_metrics_dir = os.path.join(base_dir, "fold_metrics") # Subfolder for individual fold Excels

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True) 
    os.makedirs(fold_metrics_dir, exist_ok=True)

    cms = []
    all_gt = []
    all_probs = []
    all_preds = []
    all_samples = []
    
    # Store metrics dicts for each fold
    fold_results_accumulator = []

    # --- CV Loop ---
    for fold in folds:
        print(f"\n--- FOLD {fold} ---")
        train_df = df[df[folds_col] != fold]
        val_df = df[df[folds_col] == fold]

        X_train = train_df["embeddings"].to_list()
        Y_train = train_df[label_col].astype(int).to_numpy()
        X_val = val_df["embeddings"].to_list()
        Y_val = val_df[label_col].astype(int).to_numpy()
        samples_val = val_df["SAMPLES"].to_list()

        print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

        # Model & Optimizer
        model = make_mil_model(mil_model_name, n_classes=n_classes, L=L)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Loss Function Setup
        if use_class_weights:
            cw = compute_class_weight("balanced", classes=np.unique(Y_train), y=Y_train)
            cw = torch.tensor(cw, dtype=torch.float32).to(device)
        else:
            cw = None

        if loss_name == "focal_loss":
            criterion = FocalLoss(alpha=cw, gamma=2.0, reduction="mean")
        else:
            criterion = nn.CrossEntropyLoss(weight=cw, reduction="sum")

        # Train with save_dir
        train_model(
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            scheduler=scheduler, 
            train_data=X_train, 
            train_labels=Y_train, 
            test_data=X_val, 
            test_labels=Y_val, 
            epochs=epochs, 
            run_name=f"{run_tag}_cv_k{fold}",
            save_dir=plots_dir
        )

        # Validate
        cm_k, y_true_k, y_pred_raw_k = validate_model(model, X_val, Y_val)
        cm_fixed = _standardize_confmat(y_true_k, y_pred_raw_k, class_labels)
        cms.append(cm_fixed)

        # Process probabilities
        probs_k = _ensure_probs(np.asarray(y_pred_raw_k), n_classes)
        preds_lbl_k = probs_k.argmax(axis=1)
        y_true_k_clean = _to_label_array(y_true_k)

        # --- FOLD METRICS CALCULATION (Using PROVIDED FUNCTION) ---
        fold_excel_path = os.path.join(fold_metrics_dir, f"{run_tag}_fold_{fold}_metrics.xlsx")
        
        # We pass y_true and y_score to compute AUC correctly within the function
        # The function saves the Excel and returns a DataFrame
        fold_metrics_df = compute_metrics_from_cm(
            cm=cm_fixed, 
            y_true=y_true_k_clean, 
            y_score=probs_k, 
            output_path=fold_excel_path
        )
        
        # Convert the DataFrame row to a dictionary and add the Fold number
        fold_metrics_dict = fold_metrics_df.to_dict(orient='records')[0]
        fold_metrics_dict['Fold'] = fold
        fold_results_accumulator.append(fold_metrics_dict)

        # Store aggregated results for global matrix later
        all_gt.append(y_true_k_clean)
        all_probs.append(probs_k)
        all_preds.append(preds_lbl_k)
        all_samples.extend(samples_val)

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # --- SUMMARY STATS (Mean/Std across folds) ---
    df_folds = pd.DataFrame(fold_results_accumulator)
    if not df_folds.empty:
        # Move 'Fold' to first column
        cols = ['Fold'] + [c for c in df_folds.columns if c != 'Fold']
        df_folds = df_folds[cols]
        
        # Compute Mean and Std (numeric columns only)
        numeric_df = df_folds.select_dtypes(include=[np.number])
        if 'Fold' in numeric_df: numeric_df = numeric_df.drop(columns=['Fold'])
        
        mean_series = numeric_df.mean()
        std_series = numeric_df.std()
        
        # Append summary rows
        df_folds.loc[len(df_folds)] = {'Fold': 'Mean', **mean_series.to_dict()}
        df_folds.loc[len(df_folds)] = {'Fold': 'Std', **std_series.to_dict()}
        
        summary_path = os.path.join(base_dir, f"{run_tag}_cv_fold_summary.xlsx")
        df_folds.to_excel(summary_path, index=False)
        print(f"[Done] Saved CV Summary (Mean/Std) to {summary_path}")

    # --- Aggregated Results (Global Confusion Matrix) ---
    cm_total = np.sum(np.stack(cms), axis=0)
    y_true_all = np.concatenate(all_gt)
    y_pred_probs_all = np.concatenate(all_probs, axis=0)
    y_pred_lbl_all = np.concatenate(all_preds)

    # Metrics for the Aggregated Matrix
    metrics_path = os.path.join(base_dir, f"{run_tag}__cv_aggregated_metrics.xlsx")
    compute_metrics_from_cm(cm_total, y_true=y_true_all, y_score=y_pred_probs_all, output_path=metrics_path)
    plot_confmx(cm_total, f"{run_tag}_cv", base_dir)

    # Save detailed scores
    df_scores = pd.DataFrame(y_pred_probs_all, columns=[f"prob_{c}" for c in class_labels])
    df_scores["GT"] = y_true_all
    df_scores["PRED"] = y_pred_lbl_all
    df_scores["SAMPLE"] = all_samples
    df_scores.to_excel(os.path.join(scores_dir, "cv_output_scores.xlsx"), index=False)
    print(f"[Done] Saved CV output scores to {scores_dir}")
    print(f"[Done] Saved Training curves to {plots_dir}")


# --- MAIN ---
def main():
    p = argparse.ArgumentParser(description="Modularized MIL Runner for Single Excel CV with Dual Embedding Sources.")
    
    # Paths configuration
    p.add_argument("--excel_path", type=str, 
                   default="./EXCEL_PATH_WITH_PARTITIONS.xlsx",
                   help="Path to the Excel file containing labels and fold info.")
    
    p.add_argument("--emb_dir_1", type=str, default="./FIRST_DIRECTORY_CONTAINING_EMBEDDINGS",
                   help="First directory containing embeddings.")
    p.add_argument("--emb_dir_2", type=str, default="./SECOND_DIRECTORY_CONTAINING_EMBEDDINGS",
                   help="Second directory containing embeddings.")
    
    # NOMBRE DEL MODELO DE EMBEDDING (Para estructura de carpetas)
    p.add_argument("--embedding_model", type=str, default="dinov2_base",
                   help="Name of the embedding model (used as subfolder name).")

    # Target & Training configuration
    p.add_argument("--target", type=str, choices=["ACTIVITY", "VASCULAR", "EPITHELIAL"], default="ACTIVITY",
                   help="Target variable (column name) to use as Y.")
    p.add_argument("--folds_col", type=str, default="FOLD", help="Column name indicating the fold number.")
    
    p.add_argument("--mil_model", type=str, default="MTSPMIL") 
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--classes", type=int, choices=[2, 3], default=2)
    p.add_argument("--loss", type=str, choices=["crossentropy", "focal_loss"], default="crossentropy")
    p.add_argument("--use_class_weights", type=lambda x: (str(x).lower() == "true"), default=True)
    p.add_argument("--results_root", type=str, default="./PATH_TO_RESULTS_FOLDER")
    
    args = p.parse_args()

    # 1. Load Excel Data
    print(f"[Main] Loading Excel: {args.excel_path}")
    df = load_dataframe(args.excel_path, required_cols=[args.target, args.folds_col]) # Uses utils.misc.load_dataframe

    # 2. Load and Validate Embeddings (Strict Structure)
    dirs_to_check = [args.emb_dir_1, args.emb_dir_2]
    
    df_full = load_and_validate_embeddings(
        df=df,
        embedding_dirs=dirs_to_check,
        label_col=args.target,
        folds_col=args.folds_col,
        classes=args.classes,
        embedding_model=args.embedding_model 
    )

    # 3. Create Run Tag
    tag = f"{args.target.lower()}_{args.mil_model}_cls{args.classes}_{args.loss}_{str(args.lr)}_{args.embedding_model}"

    # 4. Run Cross Validation
    run_cross_validation(
        df=df_full,
        label_col=args.target,
        folds_col=args.folds_col,
        mil_model_name=args.mil_model,
        lr=args.lr,
        epochs=args.epochs,
        use_class_weights=args.use_class_weights,
        loss_name=args.loss,
        results_root=args.results_root,
        run_tag=tag
    )

if __name__ == "__main__":
    main()