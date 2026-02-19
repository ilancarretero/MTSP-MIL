"""
Main function to extract embeddings from videos (frames)
"""
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import torch
import cv2
import numpy as np 
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModel,
    Dinov2Model
)
from open_clip import create_model_from_pretrained
from torchvision import transforms, models
from huggingface_hub import login

# Add directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

#from local_data.constants import *
#from utils.misc import set_seeds
import random
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

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seed for reproducibility
set_seeds(42, use_cuda=device == "cuda")

# HF login
login('your_hf_token_here')

class FrameEmbedder:
    """
    Wrapper for frame-level embedding extraction
    """
    def __init__(self, name, model, embed_fn):
        self.name = name
        #self.model = model.to(device).eval()
        self.model = model
        self.embed_fn = embed_fn
        
    def embed(self, frame):
        return self.embed_fn(frame, self.model, device)
    
def get_embedder(name: str) -> FrameEmbedder:
    """
    Return a FrameEmbedder for the given model
    Supported: BiomedCLIP, CONCH, microsam, dinov2_base
    """
    
    name_l = name.lower()
    
    if name_l == 'biomedclip':
        model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
        model.eval()
        def embed_fn(frame, model, device):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input = preprocess(img).to(device)
            with torch.no_grad():
                feats = model(input.unsqueeze(0))[0]
            return feats[0].cpu().numpy()
        return FrameEmbedder(name_l, model, embed_fn)
    
    elif name_l == 'conch':
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)  
        model, eval_transform = titan.return_conch()
        model.to(device)
        model.eval()
        def embed_fn(frame, model, device):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input = eval_transform(img).to(device)
            with torch.no_grad():
                feats = model(input.unsqueeze(0))
            return feats[0].cpu().numpy()
        return FrameEmbedder(name_l, model, embed_fn)

    elif name_l == 'microsam':
        # Import modules and model
        from micro_sam.util import get_sam_model, precompute_image_embeddings
        predictor = get_sam_model(
            model_type='vit_b_lm',
            device=device
        )
        predictor.model.to(device)
        predictor.model.eval()
        # Define embedding function
        def embed_fn(frame, predictor, device):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            np_img = np.array(img)
            embeddings = precompute_image_embeddings(
                predictor,
                np_img,
                save_path=None,
                ndim=2,
                tile_shape=None,
                halo=None,
                verbose=False
            )
            feats = embeddings['features']
            arr = feats if isinstance(feats, np.ndarray) else np.asarray(feats[:])
            #return arr.mean(axis=(2, 3))
            return arr.max(axis=(2, 3)) #instead of mean
        return FrameEmbedder(name_l, predictor, embed_fn)
    
    elif name_l == 'dinov2_base':
        # Load colon-pretrained DINOv2
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        model.eval()
        model.to(device)
        
        # Define transformation function
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Define embedding function
        def embed_fn(frame, model, device):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model(input)
                cls_embedding = feats.last_hidden_state[:, 0] 
            return cls_embedding[0].cpu().numpy()
        return FrameEmbedder(name_l, model, embed_fn)
              
    else:
        raise ValueError(f"Unknown model: {name}")

def process(args):
    """
    Extract frame embeddings for all .mp4 videos in args.data_root_path using
    the selected model, and save to args.data_embeddings_path.
    """
    embedder = get_embedder(args.model)
    in_dir = Path(args.data_root_path)
    out_dir = Path(args.data_embeddings_path) / embedder.name 
    out_dir.mkdir(parents=True, exist_ok=True)

    for vid_path in in_dir.glob('*.mp4'):
        print(f"Processing {vid_path.name} with {embedder.name}…")
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"Cannot open {vid_path.name}, skipping.")
            continue

        embeddings = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Optional: resize frame if models expect smaller input
            # frame = cv2.resize(frame, (224, 224))
            emb = embedder.embed(frame)
            embeddings.append(emb)
        cap.release()

        if embeddings:
            arr = np.stack(embeddings, axis=0)
            out_file = out_dir / f"{vid_path.stem}_{embedder.name}_embeddings.npy"
            np.save(out_file, arr)
            print(f"Saved embeddings {arr.shape} to {out_file.name}")
        else:
            print(f"No frames read from {vid_path.name}")


def main():
    
    parser = argparse.ArgumentParser()
    path_dataset_preprocessed = './folder_videos'
    path_dataset_embeddings = './folder_to_save_embeddings'
    # Folders, data, etc...
    parser.add_argument('--data_root_path', default=path_dataset_preprocessed)
    parser.add_argument('--data_embeddings_path', default=path_dataset_embeddings)

    # Model
    parser.add_argument('--model', default='dinov2_base',
                        choices=['BiomedCLIP', 'CONCH', 'microsam', 'dinov2_base'])
    
    args, unknown = parser.parse_known_args()
    
    process(args=args)
    
    
if __name__ == "__main__":
    main()
    
