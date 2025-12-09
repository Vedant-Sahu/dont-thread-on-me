"""
Linear Extrapolation Baseline for Community Health Prediction

Extracts temporal features from 12-month graph sequences and uses linear
regression to predict community health metrics 6 months ahead.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from torch_geometric.data import HeteroData


def extract_temporal_features(monthly_graphs: List[HeteroData]) -> np.ndarray:
    """
    Extract features from 12-month graph sequence.
    
    Features:
    - Trends (slopes over last 6 months): 4 metrics = 4
    - Means (over 12 months): 4 metrics = 4
    - Aggregated graph features (mean over 12 months): 15
    
    Total: 23 features
    
    Args:
        monthly_graphs: List of 12 HeteroData graphs
        
    Returns:
        Feature vector of shape (23,)
    """
    if len(monthly_graphs) != 12:
        raise ValueError(f"Expected 12 graphs, got {len(monthly_graphs)}")
    
    # Extract community metrics from each graph
    metrics = []
    tag_features_list = []
    user_features_list = []
    graph_sizes = []
    
    for graph in monthly_graphs:
        # Community metrics
        if hasattr(graph, 'y') and graph.y:
            metrics.append({
                'qpd': graph.y.get('qpd', 0.0),
                'answer_rate': graph.y.get('answer_rate', 0.0),
                'retention': graph.y.get('retention', 0.0),
                'growth': graph.y.get('growth', 0.0)
            })
        else:
            metrics.append({'qpd': 0.0, 'answer_rate': 0.0, 'retention': 0.0, 'growth': 0.0})
        
        # Aggregated node features
        if 'tag' in graph and graph['tag'].x is not None:
            tag_features_list.append(graph['tag'].x.mean(dim=0).numpy())
        else:
            tag_features_list.append(np.zeros(7))
        
        if 'user' in graph and graph['user'].x is not None:
            user_features_list.append(graph['user'].x.mean(dim=0).numpy())
        else:
            user_features_list.append(np.zeros(5))
        
        # Graph size statistics
        num_tags = graph['tag'].x.shape[0] if 'tag' in graph and graph['tag'].x is not None else 0
        num_users = graph['user'].x.shape[0] if 'user' in graph and graph['user'].x is not None else 0
        num_edges = 0
        if ('tag', 'cooccurs', 'tag') in graph.edge_types:
            num_edges += graph['tag', 'cooccurs', 'tag'].edge_index.shape[1]
        if ('user', 'contributes', 'tag') in graph.edge_types:
            num_edges += graph['user', 'contributes', 'tag'].edge_index.shape[1]
        graph_sizes.append([num_tags, num_users, num_edges])
    
    # Convert to numpy arrays
    metrics = np.array([[m['qpd'], m['answer_rate'], m['retention'], m['growth']] for m in metrics])
    tag_features = np.array(tag_features_list)
    user_features = np.array(user_features_list)
    graph_sizes = np.array(graph_sizes)
    
    features = []
    
    # Trends (slopes over last 6 months)
    time_indices = np.arange(6)
    for metric_idx in range(4):
        values = metrics[-6:, metric_idx]
        if len(values) > 1 and np.std(values) > 1e-6:
            slope = np.polyfit(time_indices, values, 1)[0]
        else:
            slope = 0.0
        features.append(slope)
    
    # Means (over 12 months)
    means = metrics.mean(axis=0)
    features.extend(means)
    
    # Aggregated graph features (mean over 12 months)
    mean_tag_features = tag_features.mean(axis=0)
    mean_user_features = user_features.mean(axis=0)
    mean_graph_sizes = graph_sizes.mean(axis=0)
    
    features.extend(mean_tag_features)
    features.extend(mean_user_features)
    features.extend(mean_graph_sizes)
    
    return np.array(features, dtype=np.float32)


class LinearExtrapolationBaseline:
    """
    Linear regression baseline for community health prediction.
    
    Trains separate linear models for each metric (qpd, answer_rate, retention, growth).
    Uses temporal features extracted from 12-month graph sequences.
    """
    
    def __init__(self):
        self.models = {
            'qpd': LinearRegression(),
            'answer_rate': LinearRegression(),
            'retention': LinearRegression(),
            'growth': LinearRegression()
        }
        self.scalers = {
            'qpd': StandardScaler(),
            'answer_rate': StandardScaler(),
            'retention': StandardScaler(),
            'growth': StandardScaler()
        }
        self.is_fitted = False
    
    def fit(self, train_loader, show_progress=True):
        """
        Train linear models on training data.
        
        Args:
            train_loader: DataLoader with (batch_graphs, batch_targets) tuples
            show_progress: Whether to show progress bar
        """
        from tqdm import tqdm
        
        X_train = []
        y_train = {metric: [] for metric in self.models.keys()}
        
        iterator = tqdm(train_loader, desc="Extracting features") if show_progress else train_loader
        
        for batch_graphs, batch_targets in iterator:
            for graphs in batch_graphs:
                features = extract_temporal_features(graphs)
                X_train.append(features)
            
            for metric in y_train.keys():
                y_train[metric].extend(batch_targets[metric].cpu().numpy().tolist())
        
        X_train = np.array(X_train)
        
        print(f"Training {len(self.models)} models on {len(X_train)} samples...")
        metric_iterator = tqdm(self.models.keys(), desc="Training models") if show_progress else self.models.keys()
        
        for metric in metric_iterator:
            self.scalers[metric].fit(X_train)
            X_scaled = self.scalers[metric].transform(X_train)
            self.models[metric].fit(X_scaled, y_train[metric])
        
        self.is_fitted = True
    
    def predict(self, batch_graphs: List[List[HeteroData]]) -> Dict[str, np.ndarray]:
        """
        Predict metrics for a batch of graph sequences.
        
        Args:
            batch_graphs: List of lists, each inner list contains 12 graphs
            
        Returns:
            Dictionary with predictions for each metric
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array([extract_temporal_features(graphs) for graphs in batch_graphs])
        
        predictions = {}
        for metric in self.models.keys():
            X_scaled = self.scalers[metric].transform(X)
            pred = self.models[metric].predict(X_scaled)
            predictions[metric] = pred
        
        return predictions
    
    def predict_single(self, monthly_graphs: List[HeteroData]) -> Dict[str, float]:
        """
        Predict metrics for a single graph sequence.
        
        Args:
            monthly_graphs: List of 12 HeteroData graphs
            
        Returns:
            Dictionary with single prediction for each metric
        """
        batch_predictions = self.predict([monthly_graphs])
        return {metric: float(pred[0]) for metric, pred in batch_predictions.items()}
    
    def save(self, save_path: Path):
        """Save model and scalers to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, load_path: Path):
        """Load model and scalers from disk."""
        load_path = Path(load_path)
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        baseline = cls()
        baseline.models = data['models']
        baseline.scalers = data['scalers']
        baseline.is_fitted = data['is_fitted']
        
        return baseline


def create_baseline():
    """Factory function to create baseline model."""
    return LinearExtrapolationBaseline()


def evaluate_baseline(baseline, data_loader, show_progress=True):
    """
    Evaluate baseline on a data loader.
    
    Args:
        baseline: Fitted LinearExtrapolationBaseline instance
        data_loader: DataLoader with (batch_graphs, batch_targets) tuples
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary with metrics: 'loss', 'mse', 'mae', 'r2' for each metric
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from tqdm import tqdm
    
    all_predictions = {metric: [] for metric in baseline.models.keys()}
    all_targets = {metric: [] for metric in baseline.models.keys()}
    
    iterator = tqdm(data_loader, desc="Evaluating") if show_progress else data_loader
    
    for batch_graphs, batch_targets in iterator:
        predictions = baseline.predict(batch_graphs)
        
        for metric in baseline.models.keys():
            all_predictions[metric].extend(predictions[metric])
            all_targets[metric].extend(batch_targets[metric].cpu().numpy().tolist())
    
    results = {}
    for metric in baseline.models.keys():
        pred = np.array(all_predictions[metric])
        true = np.array(all_targets[metric])
        
        results[metric] = {
            'mse': mean_squared_error(true, pred),
            'mae': mean_absolute_error(true, pred),
            'r2': r2_score(true, pred),
            'rmse': np.sqrt(mean_squared_error(true, pred))
        }
    
    # Average across metrics
    results['overall'] = {
        'mse': np.mean([results[m]['mse'] for m in baseline.models.keys()]),
        'mae': np.mean([results[m]['mae'] for m in baseline.models.keys()]),
        'r2': np.mean([results[m]['r2'] for m in baseline.models.keys()]),
        'rmse': np.mean([results[m]['rmse'] for m in baseline.models.keys()])
    }
    
    return results

