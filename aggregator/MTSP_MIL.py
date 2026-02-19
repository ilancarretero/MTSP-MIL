import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. UTILS
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the 
    tokens in the sequence. Uses standard sine and cosine functions of 
    different frequencies.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==============================================================================
# 2. TEMPORAL MULTI-SCALE AGGREGATION (TMI)
# ==============================================================================
class TemporalMultiScaleAggregation(nn.Module):
    """
    Temporal Multi-Scale Aggregation (TMI) Module.
    Captures temporal dynamics at multiple granularities (local, regional, and global)
    using parallel convolutional pathways.
    """
    def __init__(self, dim):
        super().__init__()
        # Local pathway: Preserves original frame-level features
        self.scale_local = nn.Identity()
        
        # Regional pathway: Captures short-term temporal neighborhood 
        # using dilated group convolutions for parameter efficiency
        self.scale_regional = nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        
        # Global pathway: Extracts broader temporal context
        self.scale_global = nn.AvgPool1d(kernel_size=7, stride=1, padding=3, count_include_pad=False)
        
        # Feature fusion and normalization
        self.fusion = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Transpose for 1D convolution: [Batch, Seq_len, Dim] -> [Batch, Dim, Seq_len]
        x_t = x.transpose(1, 2)
        
        s1 = x
        s2 = self.scale_regional(x_t).transpose(1, 2)
        s3 = self.scale_global(x_t).transpose(1, 2)
        
        # Concatenate multi-scale representations along the feature dimension
        combined = torch.cat([s1, s2, s3], dim=-1)
        out = self.fusion(combined)
        
        return self.norm(out)

# ==============================================================================
# 3. SEMANTIC PROTOTYPE LEARNING (SPL)
# ==============================================================================
class SemanticPrototypeLearning(nn.Module):
    """
    Semantic Prototype Learning (SPL) Module.
    Projects features into an orthogonal prototype space to enhance interpretability 
    and feature separation.
    
    Improvements:
    1. Pure normalized Cosine Similarity for robust latent space mapping.
    2. Learnable temperature scaling for calibrated attention distribution.
    3. Returns normalized prototypes for auxiliary orthogonality constraints.
    """
    def __init__(self, dim, num_prototypes=16, init_temp=0.07):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        
        # Learnable semantic prototypes
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, dim))
        
        # Learnable temperature (log-parameterized for numerical stability)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / init_temp))

    def forward(self, x):
        # x: [B, N, D]
        # L2 Normalization for magnitude-invariant pattern matching
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        
        # Compute scaled cosine similarity
        # Shape: [B, N, D] @ [1, D, K] -> [B, N, K]
        logit_scale = self.logit_scale.exp()
        sim_scores = torch.matmul(x_norm, p_norm.transpose(1, 2)) * logit_scale
        
        # Softmax over prototypes to get soft assignments
        sim_probs = torch.softmax(sim_scores, dim=-1) # [B, N, K]
        
        # Prototype-based feature reconstruction
        x_proto = torch.matmul(sim_probs, self.prototypes)
        
        # Residual connection and normalized prototype matrix export
        return x + x_proto, p_norm.squeeze(0)

# ==============================================================================
# 4. FULL ARCHITECTURE & AGGREGATOR
# ==============================================================================
class MTSP_Aggregator(nn.Module):
    """
    Main aggregator combining TMI, SPL, and GTM modules.
    """
    def __init__(self, L, hidden_dim=256, n_heads=4, dropout=0.25):
        super().__init__()
        
        # Input projection
        self.fc_in = nn.Sequential(
            nn.Linear(L, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 1. Temporal Multi-Scale Aggregation (TMI)
        self.pyramid = TemporalMultiScaleAggregation(hidden_dim)
        
        # 2. Semantic Prototype Learning (SPL)
        self.proto_layer = SemanticPrototypeLearning(hidden_dim, num_prototypes=16)
        
        # 3. Global Temporal Modeling (GTM)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Attention-based pooling (Part of GTM)
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features):
        if features.dim() == 2: features = features.unsqueeze(0)
        
        # Initial projection
        x = self.fc_in(features)
        
        # TMI: Multi-scale feature extraction
        x_multi = self.pyramid(x)
        
        # SPL: Stabilized features and prototype matrix
        x_stable, prototypes = self.proto_layer(x_multi)
        
        # GTM: Contextual encoding and attention pooling
        x_pos = self.pos_encoder(x_stable)
        x_context = self.context_encoder(x_pos)
        
        att_logits = self.attn_net(x_context)
        w = torch.softmax(att_logits, dim=1)
        embedding = torch.sum(x_context * w, dim=1).squeeze(0)
        
        return embedding, w.squeeze(), prototypes

class MTSPMIL(nn.Module):
    """
    Full Multiple Instance Learning network (MTSP-MIL).
    """
    def __init__(self, n_classes, L, p=0.25):
        super(MTSPMIL, self).__init__()
        self.hidden_dim = 256
        self.aggregator = MTSP_Aggregator(L, hidden_dim=self.hidden_dim, dropout=p)
        
        # 4. Classification
        self.classifier = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, features):
        embedding, w, prototypes = self.aggregator(features)
        # Classification head
        output = self.classifier(embedding)
        return output, embedding
    
# ==============================================================================
# 5. ABLATION EXPERIMENTS
# ==============================================================================

class AB1_OnlyAttn(nn.Module):
    """
    Ablation 1: Baseline Architecture.
    Modules used: 
    - None (No TMI, No SPL, No Transformer-based GTM).
    - Uses only basic Attention Pooling and Classification.
    """
    def __init__(self, n_classes, L, hidden_dim=256, p=0.25):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(L, hidden_dim), nn.ReLU(), nn.Dropout(p))
        
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        if features.dim() == 2: features = features.unsqueeze(0)
        x = self.fc_in(features)
        
        att_logits = self.attn_net(x)
        w = torch.softmax(att_logits, dim=1)
        
        embedding = torch.sum(x * w, dim=1).squeeze(0)
        output = self.classifier(embedding)
        return output, embedding
    

class AB2_AttnTransformer(nn.Module):
    """
    Ablation 2: Adding Global Temporal Modeling.
    Modules used:
    - Global Temporal Modeling (GTM)
    - Classification
    """
    def __init__(self, n_classes, L, hidden_dim=256, n_heads=4, p=0.25):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(L, hidden_dim), nn.ReLU(), nn.Dropout(p))
        
        # GTM
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=p, activation='gelu', batch_first=True, norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        
        # Classification
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        if features.dim() == 2: features = features.unsqueeze(0)
        x = self.fc_in(features)
        
        x = self.pos_encoder(x)
        x_context = self.context_encoder(x)
        
        att_logits = self.attn_net(x_context)
        w = torch.softmax(att_logits, dim=1)
        embedding = torch.sum(x_context * w, dim=1).squeeze(0)
        return self.classifier(embedding), embedding
    

class AB3_AttnTransProto(nn.Module):
    """
    Ablation 3: Adding Semantic Prototypes.
    Modules used:
    - Semantic Prototype Learning (SPL)
    - Global Temporal Modeling (GTM)
    - Classification
    """
    def __init__(self, n_classes, L, hidden_dim=256, n_heads=4, p=0.25):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(L, hidden_dim), nn.ReLU(), nn.Dropout(p))
        
        # SPL
        self.proto_layer = SemanticPrototypeLearning(hidden_dim, num_prototypes=16)
        
        # GTM
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        
        # Classification
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        if features.dim() == 2: features = features.unsqueeze(0)
        x = self.fc_in(features)
        
        x_stable, _ = self.proto_layer(x) 
        x_pos = self.pos_encoder(x_stable)
        x_context = self.context_encoder(x_pos)
        
        w = torch.softmax(self.attn_net(x_context), dim=1)
        embedding = torch.sum(x_context * w, dim=1).squeeze(0)
        return self.classifier(embedding), embedding
    

class AB4_AttnTransPyramid(nn.Module):
    """
    Ablation 4: Adding Temporal Multi-Scale Aggregation (No Prototypes).
    Modules used:
    - Temporal Multi-Scale Aggregation (TMI)
    - Global Temporal Modeling (GTM)
    - Classification
    """
    def __init__(self, n_classes, L, hidden_dim=256, n_heads=4, p=0.25):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(L, hidden_dim), nn.ReLU(), nn.Dropout(p))
        
        # TMI
        self.pyramid = TemporalMultiScaleAggregation(hidden_dim)
        
        # GTM
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        
        # Classification
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        if features.dim() == 2: features = features.unsqueeze(0)
        x = self.fc_in(features)
        
        x_multi = self.pyramid(x)
        x_pos = self.pos_encoder(x_multi)
        x_context = self.context_encoder(x_pos)
        
        w = torch.softmax(self.attn_net(x_context), dim=1)
        embedding = torch.sum(x_context * w, dim=1).squeeze(0)
        return self.classifier(embedding), embedding