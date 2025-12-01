"""
Temporal Graph Neural Network for Online Community Health Prediction

This module implements a heterogeneous GNN with temporal modeling to predict
engagement metrics for Stack Exchange communities 6 months ahead.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv


class TemporalCommunityGNN(nn.Module):
    """
    Temporal GNN for predicting community health trajectories.
    
    Architecture:
    1. Batch normalization on raw features
    2. Linear projection to shared embedding space
    3. Multiple HeteroConv layers with SAGEConv
    4. ReLU activation + Dropout after each conv
    5. Graph-level pooling (separate for users and tags)
    6. Stack 12 monthly embeddings into temporal sequence
    7. Transformer encoder for temporal modeling
    8. Multi-task prediction heads for 4 engagement metrics
    
    Args:
        user_feat_dim (int): Dimension of raw user features
        tag_feat_dim (int): Dimension of raw tag features
        hidden_dim (int): Hidden dimension for embeddings
        num_conv_layers (int): Number of graph convolutional layers
        num_transformer_layers (int): Number of transformer encoder layers
        num_attention_heads (int): Number of attention heads in transformer
        dropout (float): Dropout probability
        transformer_ffn_dim (int): Feedforward network dimension in transformer
    """
    
    def __init__(
        self,
        user_feat_dim: int,
        tag_feat_dim: int,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        num_transformer_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        transformer_ffn_dim: int = 256
    ):
        super(TemporalCommunityGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        
        # ===================================================================
        # Stage 1: Batch Normalization and Feature Projection
        # ===================================================================
        
        # Batch normalization before projection for stable training
        self.user_norm = nn.BatchNorm1d(user_feat_dim)
        self.tag_norm = nn.BatchNorm1d(tag_feat_dim)
        
        # Project raw features to shared embedding space
        self.user_proj = nn.Linear(user_feat_dim, hidden_dim)
        self.tag_proj = nn.Linear(tag_feat_dim, hidden_dim)
        
        # ===================================================================
        # Stage 2: Graph Convolutional Layers
        # ===================================================================
        
        # Create HeteroConv layers with SAGEConv for all edge types
        self.convs = nn.ModuleList()
        for _ in range(num_conv_layers):
            conv = HeteroConv({
                # Tag co-occurrence edges (tag -> tag)
                ("tag", "cooccurs", "tag"): 
                    SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
                
                # User contribution edges (user -> tag)
                ("user", "contributes", "tag"): 
                    SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean"),
                
                # Reverse contribution edges (tag -> user)
                ("tag", "contributed_to_by", "user"): 
                    SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr="mean"),
            }, aggr="mean")
            
            self.convs.append(conv)
        
        # Dropout for regularization after each conv layer
        self.conv_dropout = nn.Dropout(dropout)
        
        # ===================================================================
        # Stage 3: Temporal Modeling with Transformer
        # ===================================================================
        
        # Community embedding dimension = 2 * hidden_dim (user + tag pooled)
        community_emb_dim = 2 * hidden_dim
        
        # Transformer encoder for capturing temporal dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=community_emb_dim,
            nhead=num_attention_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            batch_first=True  # Input shape: [batch, seq_len, emb_dim]
        )
        
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # ===================================================================
        # Stage 4: Multi-task Prediction Heads
        # ===================================================================
        
        # Separate prediction head for each engagement metric
        self.qpd_head = nn.Linear(community_emb_dim, 1)  # Questions per day
        self.ansrate_head = nn.Linear(community_emb_dim, 1)  # Answer rate
        self.retention_head = nn.Linear(community_emb_dim, 1)  # User retention
        self.growth_head = nn.Linear(community_emb_dim, 1)  # New user growth
        
    def project_features(self, x_dict):
        """
        Stage 1: Apply batch normalization and project to shared embedding space.
        
        Args:
            x_dict (dict): Dictionary with 'user' and 'tag' node features
            
        Returns:
            dict: Projected embeddings for each node type
        """
        x_dict_projected = {}
        
        # Project user features
        if "user" in x_dict:
            x_user = x_dict["user"]
            x_user = self.user_norm(x_user)
            x_dict_projected["user"] = self.user_proj(x_user)
        
        # Project tag features
        if "tag" in x_dict:
            x_tag = x_dict["tag"]
            x_tag = self.tag_norm(x_tag)
            x_dict_projected["tag"] = self.tag_proj(x_tag)
        
        return x_dict_projected
    
    def apply_conv_layers(self, x_dict, edge_index_dict):
        """
        Stage 2: Apply graph convolutional layers with ReLU and dropout.
        
        Args:
            x_dict (dict): Node embeddings for each type
            edge_index_dict (dict): Edge indices for each relation type
            
        Returns:
            dict: Updated node embeddings after convolution
        """
        for conv in self.convs:
            # Apply heterogeneous graph convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply ReLU activation to all node types
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
            
            # Apply dropout
            x_dict = {key: self.conv_dropout(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def pool_graph(self, x_dict):
        """
        Stage 3: Pool node embeddings to create community-level representation.
        
        Separately pools user and tag embeddings, then concatenates them to
        preserve type-specific information while creating unified representation.
        
        Args:
            x_dict (dict): Node embeddings for each type
            
        Returns:
            torch.Tensor: Community embedding of shape [2 * hidden_dim]
        """
        # Mean pool users and tags separately
        user_pooled = x_dict["user"].mean(dim=0)  # [hidden_dim]
        tag_pooled = x_dict["tag"].mean(dim=0)    # [hidden_dim]
        
        # Concatenate into single community embedding
        community_emb = torch.cat([user_pooled, tag_pooled])  # [2 * hidden_dim]
        
        return community_emb
    
    def forward_single_graph(self, x_dict, edge_index_dict):
        """
        Forward pass for a single monthly graph.
        
        Args:
            x_dict (dict): Raw node features
            edge_index_dict (dict): Edge indices
            
        Returns:
            torch.Tensor: Community embedding for this month
        """
        # Stage 1: Project features
        x_dict = self.project_features(x_dict)
        
        # Stage 2: Apply graph convolutions
        x_dict = self.apply_conv_layers(x_dict, edge_index_dict)
        
        # Stage 3: Pool to community embedding
        community_emb = self.pool_graph(x_dict)
        
        return community_emb
    
    def forward_temporal_sequence(self, monthly_embeddings):
        """
        Stage 4: Apply temporal modeling with Transformer.
        
        Args:
            monthly_embeddings (torch.Tensor): Stacked monthly embeddings
                                               Shape: [batch_size, 12, 2*hidden_dim]
        
        Returns:
            torch.Tensor: Temporal representation, shape [batch_size, 2*hidden_dim]
        """
        # Apply transformer encoder
        temporal_out = self.temporal_encoder(monthly_embeddings)  # [batch, 12, 2*hidden]
        
        # Use final timestep as community temporal representation
        final_repr = temporal_out[:, -1, :]  # [batch, 2*hidden_dim]
        
        return final_repr
    
    def predict_metrics(self, temporal_repr):
        """
        Stage 5: Multi-task prediction of engagement metrics.
        
        Args:
            temporal_repr (torch.Tensor): Temporal representation
                                          Shape: [batch_size, 2*hidden_dim]
        
        Returns:
            dict: Predicted metrics for each task
        """
        predictions = {
            "qpd": self.qpd_head(temporal_repr).squeeze(-1),          # Questions per day
            "answer_rate": self.ansrate_head(temporal_repr).squeeze(-1),  # Answer rate
            "retention": self.retention_head(temporal_repr).squeeze(-1),  # User retention
            "growth": self.growth_head(temporal_repr).squeeze(-1)         # New user growth
        }
        
        return predictions
    
    def forward(self, monthly_graphs):
        """
        Full forward pass through the temporal GNN.
        
        Args:
            monthly_graphs (list): List of 12 monthly graphs, where each graph is a tuple:
                                  (x_dict, edge_index_dict) or HeteroData object
        
        Returns:
            dict: Predicted engagement metrics at t+6
        """
        # Process each monthly graph to get community embeddings
        monthly_embeddings = []
        
        for graph in monthly_graphs:
            # Handle both tuple and HeteroData formats
            if isinstance(graph, tuple):
                x_dict, edge_index_dict = graph
            else:
                # Assume HeteroData object
                x_dict = graph.x_dict
                edge_index_dict = graph.edge_index_dict
            
            # Get community embedding for this month
            community_emb = self.forward_single_graph(x_dict, edge_index_dict)
            monthly_embeddings.append(community_emb)
        
        # Stack into temporal sequence [batch_size, 12, 2*hidden_dim]
        # For single sample: [1, 12, 2*hidden_dim]
        monthly_embeddings = torch.stack(monthly_embeddings).unsqueeze(0)
        
        # Apply temporal modeling
        temporal_repr = self.forward_temporal_sequence(monthly_embeddings)
        
        # Predict engagement metrics
        predictions = self.predict_metrics(temporal_repr)
        
        return predictions


class TemporalCommunityGNNBatched(TemporalCommunityGNN):
    """
    Batched version of TemporalCommunityGNN for efficient training.
    
    Handles multiple communities simultaneously by processing batches of
    temporal graph sequences.
    """
    
    def forward(self, batch_monthly_graphs):
        """
        Forward pass for batched temporal sequences.
        
        Args:
            batch_monthly_graphs (list): List of lists, where each inner list
                                        contains 12 monthly graphs for one community
                                        Shape: [batch_size][12 monthly graphs]
        
        Returns:
            dict: Predicted engagement metrics for all communities in batch
        """
        batch_temporal_embeddings = []
        
        # Process each community's temporal sequence
        for community_graphs in batch_monthly_graphs:
            monthly_embeddings = []
            
            # Process each month in the sequence
            for graph in community_graphs:
                # Handle both tuple and HeteroData formats
                if isinstance(graph, tuple):
                    x_dict, edge_index_dict = graph
                else:
                    x_dict = graph.x_dict
                    edge_index_dict = graph.edge_index_dict
                
                # Get community embedding for this month
                community_emb = self.forward_single_graph(x_dict, edge_index_dict)
                monthly_embeddings.append(community_emb)
            
            # Stack monthly embeddings for this community
            community_temporal_emb = torch.stack(monthly_embeddings)  # [12, 2*hidden_dim]
            batch_temporal_embeddings.append(community_temporal_emb)
        
        # Stack all communities: [batch_size, 12, 2*hidden_dim]
        batch_temporal_embeddings = torch.stack(batch_temporal_embeddings)
        
        # Apply temporal modeling to batch
        temporal_repr = self.forward_temporal_sequence(batch_temporal_embeddings)
        
        # Predict engagement metrics for batch
        predictions = self.predict_metrics(temporal_repr)
        
        return predictions


def create_model(
    user_feat_dim: int = 5,
    tag_feat_dim: int = 7,
    hidden_dim: int = 128,
    num_conv_layers: int = 3,
    num_transformer_layers: int = 3,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    transformer_ffn_dim: int = 256,
    batched: bool = True
):
    """
    Factory function to create temporal GNN model.
    
    Args:
        user_feat_dim (int): Number of user features
        tag_feat_dim (int): Number of tag features
        hidden_dim (int): Hidden dimension size
        num_conv_layers (int): Number of graph conv layers
        num_transformer_layers (int): Number of transformer layers
        num_attention_heads (int): Number of attention heads
        dropout (float): Dropout rate
        transformer_ffn_dim (int): Transformer FFN dimension
        batched (bool): Whether to use batched version
    
    Returns:
        nn.Module: Temporal GNN model
    """
    if batched:
        return TemporalCommunityGNNBatched(
            user_feat_dim=user_feat_dim,
            tag_feat_dim=tag_feat_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            transformer_ffn_dim=transformer_ffn_dim
        )
    else:
        return TemporalCommunityGNN(
            user_feat_dim=user_feat_dim,
            tag_feat_dim=tag_feat_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            transformer_ffn_dim=transformer_ffn_dim
        )


if __name__ == "__main__":
    # Example usage
    print("Creating Temporal Community GNN...")
    
    # Model configuration
    model = create_model(
        user_feat_dim=5,  # reputation, tenure, activity, expertise_entropy, retention
        tag_feat_dim=7,   # post_popularity, comment_popularity, avg_views, answer_quality, difficulty, diversity, growth_rate
        hidden_dim=128,
        num_conv_layers=3,
        num_transformer_layers=3,
        num_attention_heads=4,
        dropout=0.1,
        batched=True
    )
    
    print(f"\nModel Architecture:")
    print(f"  User features: 5 -> Embedding: 128")
    print(f"  Tag features: 5 -> Embedding: 128")
    print(f"  Graph conv layers: 3")
    print(f"  Transformer layers: 3 (4 attention heads)")
    print(f"  Community embedding: 256 (128 user + 128 tag)")
    print(f"  Output: 4 metrics (QPD, Answer Rate, Retention, Growth)")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")