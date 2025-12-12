"""
Graph Neural Network for Online Community Health Prediction

This module implements a heterogeneous GNN with to predict engagement
metrics for Stack Exchange communities 6 months ahead. This model
does not have a temporal component and is used as a baseline.

Optimized Version: Uses batched graph processing for 10-50× speedup.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import Batch


class CommunityGNN(nn.Module):
    """
    GNN for predicting community health trajectories.
    
    Architecture:
    1. Batch normalization on raw features
    2. Linear projection to shared embedding space
    3. Multiple HeteroConv layers with SAGEConv
    4. ReLU activation + Dropout after each conv
    5. Graph-level pooling (separate for users and tags)
    6. Average 12 monthly embeddings
    7. Multi-task prediction heads for 3 engagement metrics
    
    Args:
        user_feat_dim (int): Dimension of raw user features
        tag_feat_dim (int): Dimension of raw tag features
        hidden_dim (int): Hidden dimension for embeddings
        num_conv_layers (int): Number of graph convolutional layers
        dropout (float): Dropout probability
        use_mlp (bool): Whether to add MLP after temporal aggregation
    """
    
    def __init__(
        self,
        user_feat_dim: int,
        tag_feat_dim: int,
        hidden_dim: int = 128,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        use_mlp: bool = True
    ):
        super(CommunityGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        self.use_mlp = use_mlp
        
        # ===================================================================
        # Stage 1: Batch Normalization and Feature Projection
        # ===================================================================
        
        # Batch normalization before projection for stable training
        self.user_norm = nn.LayerNorm(user_feat_dim)
        self.tag_norm = nn.LayerNorm(tag_feat_dim)
        
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
        # Stage 3: Optional MLP for Better GPU Utilization
        # ===================================================================
        
        community_emb_dim = 2 * hidden_dim
        
        if use_mlp:
            # MLP to add computational complexity and improve GPU utilization
            self.mlp = nn.Sequential(
                nn.Linear(community_emb_dim, community_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(community_emb_dim, community_emb_dim)
            )
        
        # ===================================================================
        # Stage 4: Multi-task Prediction Heads
        # ===================================================================
        
        # Separate prediction head for each engagement metric
        self.qpd_head = nn.Linear(community_emb_dim, 1)  # Questions per day
        self.ansrate_head = nn.Linear(community_emb_dim, 1)  # Answer rate
        self.retention_head = nn.Linear(community_emb_dim, 1)  # User retention
        
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
    
    def predict_metrics(self, final_repr):
        """
        Multi-task prediction of engagement metrics.
        
        Args:
            final_repr (torch.Tensor): Final representation
                                        Shape: [batch_size, 2*hidden_dim]
        
        Returns:
            dict: Predicted metrics for each task
        """
        predictions = {
            "qpd": self.qpd_head(final_repr).squeeze(-1),          # Questions per day
            "answer_rate": self.ansrate_head(final_repr).squeeze(-1),  # Answer rate
            "retention": self.retention_head(final_repr).squeeze(-1),  # User retention
        }
        
        return predictions
    
    def forward(self, monthly_graphs):
        """
        Full forward pass through the GNN.
        
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
        
        # Stack into temporal sequence [1, 12, 2*hidden_dim]
        monthly_embeddings = torch.stack(monthly_embeddings).unsqueeze(0)
        
        # Take mean of monthly embeddings
        final_repr = monthly_embeddings.mean(dim=1)  # [1, 2*hidden_dim]
        
        # Optional MLP for better GPU utilization
        if self.use_mlp:
            final_repr = self.mlp(final_repr)
        
        # Predict engagement metrics
        predictions = self.predict_metrics(final_repr)
        
        return predictions


class CommunityGNNBatched(CommunityGNN):
    """
    Optimized batched version with parallel graph processing.
    
    Uses PyG's Batch class to process all graphs simultaneously,
    providing 10-50× speedup over sequential processing.
    """
    
    def batch_pool_graphs(self, x_dict, batch_dict, num_graphs):
        """
        Pool node embeddings for multiple graphs in parallel.
        
        Args:
            x_dict (dict): Node embeddings for all graphs combined
            batch_dict (dict): Batch assignment for each node type
            num_graphs (int): Number of graphs in the batch
            
        Returns:
            torch.Tensor: Community embeddings, shape [num_graphs, 2*hidden_dim]
        """
        community_embs = []
        
        for graph_idx in range(num_graphs):
            # Create masks for this graph
            user_mask = batch_dict["user"] == graph_idx
            tag_mask = batch_dict["tag"] == graph_idx
            
            # Pool users and tags separately for this graph
            user_pooled = x_dict["user"][user_mask].mean(dim=0)  # [hidden_dim]
            tag_pooled = x_dict["tag"][tag_mask].mean(dim=0)     # [hidden_dim]
            
            # Concatenate
            community_emb = torch.cat([user_pooled, tag_pooled])  # [2*hidden_dim]
            community_embs.append(community_emb)
        
        return torch.stack(community_embs)  # [num_graphs, 2*hidden_dim]
    
    def forward(self, batch_monthly_graphs):
        """
        Optimized forward pass with batched graph processing.
        
        Args:
            batch_monthly_graphs (list): List of lists, where each inner list
                                        contains 12 monthly graphs for one community
                                        Shape: [batch_size][12 monthly graphs]
        
        Returns:
            dict: Predicted engagement metrics for all communities in batch
        """
        batch_size = len(batch_monthly_graphs)
        seq_len = len(batch_monthly_graphs[0])  # Should be 12
        
        # =====================================================================
        # STEP 1: Flatten all graphs into a single list
        # =====================================================================
        # From: [batch_size][12 graphs] → [batch_size * 12 graphs]
        all_graphs = []
        for community_graphs in batch_monthly_graphs:
            all_graphs.extend(community_graphs)  

        # Remove unnecessary attributes
        for graph in all_graphs:
            if hasattr(graph['user'], 'user_to_idx'):
                delattr(graph['user'], 'user_to_idx')
            if hasattr(graph['tag'], 'tag_to_idx'):
                delattr(graph['tag'], 'tag_to_idx')
        
        # Total graphs to process: batch_size * 12
        total_graphs = len(all_graphs)
        
        # =====================================================================
        # STEP 2: Batch all graphs together for PARALLEL processing
        # =====================================================================
        # This is the key optimization - PyG will process all graphs at once!
        batched_data = Batch.from_data_list(all_graphs)
        
        # =====================================================================
        # STEP 3: Process ALL graphs in ONE forward pass (FAST!)
        # =====================================================================
        # Project features
        x_dict = self.project_features(batched_data.x_dict)
        
        # Apply graph convolutions (operates on all graphs simultaneously)
        x_dict = self.apply_conv_layers(x_dict, batched_data.edge_index_dict)
        
        # =====================================================================
        # STEP 4: Pool each graph separately using batch indices
        # =====================================================================
        # batched_data.batch contains indices indicating which graph each node belongs to
        batch_dict = {
            "user": batched_data["user"].batch,
            "tag": batched_data["tag"].batch
        }
        
        # Pool all graphs at once
        all_community_embs = self.batch_pool_graphs(x_dict, batch_dict, total_graphs)
        # Shape: [batch_size * 12, 2*hidden_dim]
        
        # =====================================================================
        # STEP 5: Reshape and aggregate temporal sequences
        # =====================================================================
        # Reshape: [batch_size * 12, emb_dim] → [batch_size, 12, emb_dim]
        monthly_embeddings = all_community_embs.view(batch_size, seq_len, -1)
        
        # Average over time dimension
        final_repr = monthly_embeddings.mean(dim=1)  # [batch_size, 2*hidden_dim]
        
        # Optional MLP for better GPU utilization
        if self.use_mlp:
            final_repr = self.mlp(final_repr)
        
        # =====================================================================
        # STEP 6: Predict engagement metrics
        # =====================================================================
        predictions = self.predict_metrics(final_repr)
        
        return predictions


def create_model(
    user_feat_dim: int = 5,
    tag_feat_dim: int = 7,
    hidden_dim: int = 128,
    num_conv_layers: int = 3,
    dropout: float = 0.1,
    use_mlp: bool = False,
    batched: bool = True
):
    """
    Factory function to create GNN model.
    
    Args:
        user_feat_dim (int): Number of user features
        tag_feat_dim (int): Number of tag features
        hidden_dim (int): Hidden dimension size
        num_conv_layers (int): Number of graph conv layers
        dropout (float): Dropout rate
        use_mlp (bool): Add MLP after temporal aggregation for better GPU utilization
        batched (bool): Whether to use optimized batched version (RECOMMENDED)
    
    Returns:
        nn.Module: GNN model
    """
    if batched:
        return CommunityGNNBatched(
            user_feat_dim=user_feat_dim,
            tag_feat_dim=tag_feat_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            use_mlp=use_mlp
        )
    else:
        return CommunityGNN(
            user_feat_dim=user_feat_dim,
            tag_feat_dim=tag_feat_dim,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            use_mlp=use_mlp
        )


if __name__ == "__main__":
    # Example usage
    print("Creating Optimized Community GNN...")
    print("=" * 70)
    
    # Model configuration
    model = create_model(
        user_feat_dim=5,  # reputation, tenure, activity, expertise_entropy, retention
        tag_feat_dim=7,   # post_popularity, comment_popularity, avg_views, answer_quality, difficulty, diversity, growth_rate
        hidden_dim=128,
        num_conv_layers=3,
        dropout=0.1,
        use_mlp=True,  # Set to True for better GPU utilization
        batched=True    # ALWAYS use True for training
    )
    
    print(f"\nModel Architecture:")
    print(f"  User features: 5 → Embedding: 128")
    print(f"  Tag features: 7 → Embedding: 128")
    print(f"  Graph conv layers: 3")
    print(f"  Community embedding: 256 (128 user + 128 tag)")
    print(f"  MLP enabled: True")
    print(f"  Output: 3 metrics (QPD, Answer Rate, Retention)")
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
