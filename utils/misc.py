
import os
import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score


# General functions
def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def shuffle_data(data, labels):
    combined_data = list(zip(data, labels))
    random.shuffle(combined_data)
    data, labels = zip(*combined_data)
    data, labels = list(data), np.array(labels)
    return data, labels  

# Plots functions
def plot_confmx(conf_matrix, run_name, base_results_dir, subset=None):
    TP = np.diag(conf_matrix)
    FN = np.sum(conf_matrix, axis=1) - TP
    bal_acc = np.mean(TP / (TP + FN))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('BALACC = ' + str(round(bal_acc, 4)))
    if subset is None:
        plt.savefig(os.path.join(base_results_dir, "cfmx_" + run_name + '.png'))
    else:
        plt.savefig(os.path.join(base_results_dir, "cfmx_" + run_name + '_' + str(subset) + '.png'))

def plot_figures(train_acc_epoch, test_acc_epoch, train_loss_epoch, test_loss_epoch, run_name, save_dir):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Plotting training and testing losses
    axs[0].plot(train_loss_epoch, label='Train Loss')
    axs[0].plot(test_loss_epoch, label='Test Loss')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title(f"Run: {run_name}")

    # Plotting training and testing accuracies
    axs[1].plot(train_acc_epoch, label='Train ACC')
    axs[1].plot(test_acc_epoch, label='Test ACC')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "curve_" + run_name + ".png")
        plt.savefig(save_path)
        print(f"Curve saved at: {save_path}")
    
    plt.close(fig) # Importante cerrar la figura para liberar memoria

# Load functions        
def load_dataframe(df_path: str, required_cols: list) -> pd.DataFrame:
    """
    Load the Excel file and select the required columns.
    Expected columns: Video, TotalScore, Fold.
    """
    df = pd.read_excel(df_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {missing}")
    return df #df[required_cols].copy()

def load_embeddings(df: pd.DataFrame, col_name: str, embeddings_base: str, model_name: str, less_frames: bool) -> pd.DataFrame:
    """
    For each video in the dataframe, load its embeddings (.npy file) and append as an array.
    The expected file pattern is:
      <embeddings_base>/<model_name>/<video>_<model_name>_embeddings.npy

    Returns a new DataFrame with an added 'embeddings' column.
    """
    embeddings_list = []
    for idx, row in df.iterrows():
        video = row[col_name]
        emb_file = os.path.join(
            embeddings_base,
            model_name,
            f"{video}_{model_name}_embeddings.npy"
        )
        if not os.path.isfile(emb_file):
            #continue # CHECK
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")
        emb = np.load(emb_file)
        if less_frames:
            emb = emb[::4]
        embeddings_list.append(emb)

    df_with_emb = df.copy()
    df_with_emb['embeddings'] = embeddings_list
    return df_with_emb

# Train-Val partitions from .xlsx values
def get_fold_data(
    df: pd.DataFrame,
    fold: int,
    labels: str
):
    """
    Split data into training and validation sets for a given fold.

    Returns:
      X_train: list of np.ndarray (embeddings for each video)
      Y_train: list of labels
      X_val: list of np.ndarray
      Y_val: list of labels
    """
    train_df = df[df['FOLDS'] != fold]
    val_df = df[df['FOLDS'] == fold]

    X_train = train_df['embeddings'].tolist()
    Y_train = train_df[labels].tolist()
    X_val = val_df['embeddings'].tolist()
    Y_val = val_df[labels].tolist()
    return X_train, Y_train, X_val, Y_val
        
        
# Results function
def compute_metrics_from_cm(cm, y_true=None, y_score=None, average='macro', output_path='metrics.xlsx'):
    """
    Compute classification metrics from a confusion matrix (2 or more classes) and optionally AUC from true labels and scores.
    
    Parameters:
    - cm: numpy.ndarray, shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] is count of true class i predicted as j.
    - y_true: array-like of shape (n_samples,), optional
        True labels (required for AUC).
    - y_score: array-like of shape (n_samples, n_classes) or (n_samples,), optional
        Predicted scores or probabilities for computing AUC.
    - average: str, default='macro'
        Averaging method for multiclass AUC ('macro', 'weighted', etc.).
    - output_path: str, default='metrics.xlsx'
        File path for the output Excel file.
    
    Returns:
    - pandas.DataFrame
        DataFrame with one row containing ACC, BACC, SEN, SPE, PPV, NPV, F1, AUC.
    """

    cm = np.array(cm)
    n_classes = cm.shape[0]
    # True Positives, False Positives, False Negatives, True Negatives per class
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Per-class metrics
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)
    acc_per_class = TP  / (TP + FN) 

    # Aggregated metrics
    ACC = TP.sum() / cm.sum()
    BACC = np.mean(acc_per_class)
    if np.unique(y_true).shape[0] == 2:
        # For binary classification, use the first class
        SEN = np.mean(sensitivity[1])
        SPE = np.mean(specificity[1])
        PPV = np.mean(ppv[1])
        NPV = np.mean(npv[1])
        F1 = np.mean(f1[1])
    else:
        SEN = np.mean(sensitivity)
        SPE = np.mean(specificity)
        PPV = np.mean(ppv)
        NPV = np.mean(npv)
        F1 = np.mean(f1)

    # AUC (requires y_true and y_score)
    if y_true is not None and y_score is not None:
        try:
            if np.unique(y_true).shape[0] == 2:
                y_score = y_score[:, 1]
            auc_val = roc_auc_score(y_true, y_score, multi_class='ovr', average=average)
        except Exception:
            auc_val = np.nan
    else:
        auc_val = np.nan

    metrics = {
        'ACC': [ACC],
        'BACC': [BACC],
        'SEN': [SEN],
        'SPE': [SPE],
        'PPV': [PPV],
        'NPV': [NPV],
        'F1': [F1],
        'AUC': [auc_val]
    }

    df = pd.DataFrame(metrics)
    df.to_excel(output_path, index=False)
    return df


def _to_label_array(y):
    """
    Ensure predictions/targets are 1D integer label arrays.
    If y are probabilities/logits (2D), converts via argmax.
    """
    y = np.asarray(y)
    if y.ndim > 1:
        return y.argmax(axis=1).astype(int)
    return y.astype(int)

def _standardize_confmat(y_true, y_pred, class_labels):
    """
    Build a confusion matrix with a fixed label set so that the shape is always len(class_labels) x len(class_labels).
    """
    y_true = _to_label_array(y_true)
    y_pred = _to_label_array(y_pred)
    return confusion_matrix(y_true, y_pred, labels=class_labels)
        