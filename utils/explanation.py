import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from decord import VideoReader, cpu

# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================
def find_video_path(sample_id, video_dirs):
    """
    Locates the corresponding .mp4 video file across multiple candidate directories.
    
    Args:
        sample_id (str): Unique identifier for the video sample.
        video_dirs (list): List of directory paths to search.
        
    Returns:
        str: Absolute path to the video file, or None if not found.
    """
    filename = f"{sample_id}.mp4"
    for directory in video_dirs:
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
    return None

# ==============================================================================
# 2. FRAME EXTRACTION PIPELINE
# ==============================================================================
def extract_and_save_frames(video_path, frames_indices, output_dir, prefix="frame", sampling_rate=1):
    """
    Efficiently extracts specific frames from a video using the DECORD library.
    Optimized for sparse, random-access frame retrieval to mitigate I/O bottlenecks.
    
    Args:
        video_path (str): Path to the source video.
        frames_indices (list): Target frame indices retrieved from the attention/prototype mechanisms.
        output_dir (str): Destination directory for the extracted frames.
        prefix (str): Prefix for the saved image files.
        sampling_rate (int): Stride used during initial feature extraction mapping.
    """
    if not frames_indices:
        return

    # Ensure absolute path and valid output directory
    video_path = os.path.abspath(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Decord VideoReader context
    # cpu(0) is used to load data into system RAM. gpu(0) can be used for VRAM acceleration if available.
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"[Error] Decord failed to read: {os.path.basename(video_path)}")
        return

    total_frames = len(vr)

    # Map model indices to actual video frame indices based on the sampling rate
    real_indices = [int(idx * sampling_rate) for idx in frames_indices]
    
    # Filter out-of-bound indices robustly
    valid_indices = sorted(list(set([x for x in real_indices if x < total_frames])))

    if not valid_indices:
        print(f"[Warn] Out-of-bounds indices for {os.path.basename(video_path)}")
        return

    # Batch extraction of frames: returns shape (N, H, W, 3) in RGB format
    try:
        video_frames = vr.get_batch(valid_indices).asnumpy()
    except Exception as e:
        print(f"[Error] Batch extraction failed for {os.path.basename(video_path)}")
        return

    # Process and save each extracted frame
    for i, frame_idx in enumerate(valid_indices):
        img_rgb = video_frames[i]
        
        # Color space conversion: Decord decodes to RGB, but OpenCV expects BGR for writing
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        save_name = f"{prefix}_{frame_idx}.jpg"
        save_path = os.path.join(output_dir, save_name)
        
        cv2.imwrite(save_path, img_bgr)

# ==============================================================================
# 3. INTERPRETABILITY AND EXPLANATION GENERATION
# ==============================================================================
def generate_fold_explanations(model, val_df, video_dirs, output_root, fold, device, label_col, top_k=5):
    """
    Generates quantitative and qualitative visual explanations for the MTSPMIL model.
    Extracts high-attention frames and semantic prototypes, saving both images and metric logs.
    
    Args:
        model (nn.Module): The trained model architecture.
        val_df (pd.DataFrame): Validation dataframe containing sample metadata and embeddings.
        video_dirs (list): Source directories for the raw videos.
        output_root (str): Root directory for saving explanations.
        fold (int): Current cross-validation fold identifier.
        device (torch.device): Execution device (CPU/GPU).
        label_col (str): Column name for the ground truth labels.
        top_k (int): Number of top responding frames to extract per prototype/attention mechanism.
    """
    print(f"\n[Explanation] Generating video explanations for FOLD {fold}...")
    
    fold_base_dir = os.path.join(output_root, f"fold_{fold}_val", "videos")
    os.makedirs(fold_base_dir, exist_ok=True)
    
    model.eval()
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Explaining videos"):
        sample_id = row["SAMPLES"]
        ground_truth = int(row[label_col])
        
        # 1. Video Path Resolution
        video_path = find_video_path(sample_id, video_dirs)
        
        # 2. Feature Loading & formatting: [1, Seq_len, Dim]
        features = row["embeddings"].to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        with torch.no_grad():
            # 3. Forward Pass through the Aggregator
            embedding, w, prototypes = model.aggregator(features)
            
            # --- CLASSIFICATION HEAD ---
            if embedding.dim() == 1: 
                embedding_cls = embedding.unsqueeze(0)
            else: 
                embedding_cls = embedding
            
            logits = model.classifier(embedding_cls)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0, prediction].item()
            
            # --- LATENT SPACE ACTIVATION ANALYSIS ---
            # Reconstruct the feature projections just before prototype matching
            x_proj = model.aggregator.fc_in(features)
            x_multi = model.aggregator.pyramid(x_proj)
            x_latent = x_multi.squeeze(0) # Shape: [N, D]
            
            # Compute similarity matrix (S): L2 Normalized Cosine Similarity
            x_norm = F.normalize(x_latent, dim=-1)
            p_norm = F.normalize(prototypes, dim=-1)
            sim_matrix = torch.matmul(x_norm, p_norm.T) # Shape: [N, K] where K = num_prototypes
            
            # Extract basic statistical metrics across the temporal dimension
            p_mean = torch.mean(sim_matrix, dim=0).cpu().numpy()
            p_max = torch.max(sim_matrix, dim=0).values.cpu().numpy()
            
            # Weighted Activation Score: Integrates GTM attention weights with SPL prototype similarities
            # Yields a robust metric for global prototype importance per video instance
            w_col = w.unsqueeze(1) 
            p_wtd = torch.sum(sim_matrix * w_col, dim=0).cpu().numpy()
            
            # --- ATTENTION MECHANISM EVALUATION (TOP FRAMES) ---
            actual_k = min(top_k, w.shape[0])
            top_attn_vals, top_attn_indices = torch.topk(w, k=actual_k)
            top_attn_vals = top_attn_vals.cpu().numpy()
            top_attn_indices = top_attn_indices.cpu().numpy()

            # --- VISUAL EXPLANATION EXTRACTION ---
            if video_path is not None:
                video_output_dir = os.path.join(fold_base_dir, sample_id)
                
                # Extract Global Temporal Modeling (GTM) Attention Frames
                extract_and_save_frames(video_path, top_attn_indices.tolist(), 
                                      os.path.join(video_output_dir, "imp_frames"), 
                                      prefix="imp", sampling_rate=1)
                
                # Extract Semantic Prototype Learning (SPL) Activating Frames
                proto_dir_root = os.path.join(video_output_dir, "prototypes")
                for k in range(prototypes.shape[0]):
                    # Retrieve the top-K frames that highly activate prototype k
                    _, p_indices = torch.topk(sim_matrix[:, k], k=actual_k)
                    p_indices = p_indices.cpu().numpy().tolist()
                    
                    extract_and_save_frames(video_path, p_indices, 
                                          os.path.join(proto_dir_root, f"proto_{k}"), 
                                          prefix=f"p{k}", sampling_rate=1)
            else:
                video_output_dir = os.path.join(fold_base_dir, sample_id)
                os.makedirs(video_output_dir, exist_ok=True)

            # --- QUANTITATIVE METRICS LOGGING ---
            class_dir = os.path.join(video_output_dir, "classification")
            os.makedirs(class_dir, exist_ok=True)

            # Log 1: General Classification & Full Prototype Metrics
            data_cls = {
                "Sample": [sample_id],
                "Ground_Truth": [ground_truth],
                "Prediccion": [prediction],
                "Probabilidad": [confidence]
            }
            for k in range(len(p_mean)):
                data_cls[f"P{k}_Mean"] = [p_mean[k]]
                data_cls[f"P{k}_Max"] = [p_max[k]]
                data_cls[f"P{k}_Wtd"] = [p_wtd[k]]
            
            pd.DataFrame(data_cls).to_excel(os.path.join(class_dir, "classification_info.xlsx"), index=False)
            
            # Log 2: GTM Attention Rankings
            data_attn = {
                "Rank": range(1, len(top_attn_indices) + 1),
                "Frame_Index": top_attn_indices,
                "Attention_Weight": top_attn_vals
            }
            pd.DataFrame(data_attn).to_excel(os.path.join(class_dir, "top_frames_metrics.xlsx"), index=False)

            # Log 3: SPL Top-5 Prototype Activation Rankings
            # Aggregates and sorts prototypes based on their attention-weighted relevance
            all_protos = []
            for k in range(len(p_wtd)):
                all_protos.append({
                    "Folder_Name": f"proto_{k}", 
                    "Weighted_Score": p_wtd[k],  # Primary sorting criterion
                    "Max_Score": p_max[k],
                    "Mean_Score": p_mean[k]
                })
            
            # Sort descending by weighted relevance score
            sorted_protos = sorted(all_protos, key=lambda x: x["Weighted_Score"], reverse=True)
            top_5_protos = sorted_protos[:5]
            
            # Format ranked prototype data for Excel export
            data_top_proto = {}
            for i, p_info in enumerate(top_5_protos):
                rank = i + 1
                data_top_proto[f"Rank_{rank}_Name"] = [p_info["Folder_Name"]]
                data_top_proto[f"Rank_{rank}_Wtd_Activation"] = [p_info["Weighted_Score"]]
                data_top_proto[f"Rank_{rank}_Max_Activation"] = [p_info["Max_Score"]]
            
            pd.DataFrame(data_top_proto).to_excel(os.path.join(class_dir, "top_prototypes.xlsx"), index=False)