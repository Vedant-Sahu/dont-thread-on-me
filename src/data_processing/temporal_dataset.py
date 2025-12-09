"""
Temporal dataset and dataloader for community health prediction.

Provides Dataset class for loading 12-month graph sequences with 6-month ahead targets,
and custom collate function for batching variable-size graphs.

Usage:
    from src.data_processing.temporal_dataset import TemporalCommunityDataset, collate_temporal_batch
    
    train_dataset = TemporalCommunityDataset(graph_dir='data/processed/graphs', split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_temporal_batch)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta


class TemporalCommunityDataset(Dataset):
    """
    Dataset for temporal community health prediction.
    
    Each sample consists of:
    - 12 monthly graphs (t-11 to t) as input sequence
    - Target metrics from graph at t+6 (6 months ahead)
    
    Temporal splits:
    - Train: 2008-01 to 2020-06 (can predict up to 2020-12)
    - Val: 2020-07 to 2022-09 (can predict up to 2023-03)
    - Test: 2022-10 to 2023-09 (can predict up to 2024-03)
    
    Note: Data only available until 2024-03, so we can only use sequences
    ending up to 2023-09 (to predict 2024-03).
    """
    
    def __init__(
        self,
        graph_dir: Path,
        split: str = 'train',
        sequence_length: int = 12,
        prediction_horizon: int = 6,
        min_graphs_required: int = None
    ):
        """
        Args:
            graph_dir: Root directory containing community folders with .pt files
            split: 'train', 'val', or 'test'
            sequence_length: Number of months in input sequence (default: 12)
            prediction_horizon: Months ahead to predict (default: 6)
            min_graphs_required: Minimum graphs needed per community (default: sequence_length + prediction_horizon)
        """
        self.graph_dir = Path(graph_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.min_graphs_required = min_graphs_required or (sequence_length + prediction_horizon)
        
        # Define temporal splits
        # Note: End months are chosen so that t+6 stays within data availability (up to 2024-03)
        self.split_ranges = {
            'train': ('2008-01', '2020-06'),   # Targets: 2008-07 to 2020-12
            'val': ('2020-07', '2022-09'),     # Targets: 2021-01 to 2023-03
            'test': ('2022-10', '2023-09')     # Targets: 2023-04 to 2024-03
        }
        
        # Build index of valid samples
        self.samples = self._build_sample_index()
        
        print(f"\n{split.upper()} Dataset:")
        print(f"  Date range: {self.split_ranges[split][0]} to {self.split_ranges[split][1]}")
        print(f"  Total samples: {len(self.samples)}")
        if self.samples:
            communities = set(s['community'] for s in self.samples)
            print(f"  Communities: {len(communities)}")
            print(f"  Avg samples per community: {len(self.samples) / len(communities):.1f}")
    
    def _build_sample_index(self) -> List[Dict]:
        """
        Build index of all valid (community, month) samples for this split.
        
        A sample is valid if:
        1. We have 12 consecutive months of history (t-11 to t)
        2. We have target at t+6
        3. Month t is within the split range
        4. All months form a consecutive sequence
        """
        samples = []
        start_month, end_month = self.split_ranges[self.split]
        
        # Iterate through all community directories
        for community_dir in sorted(self.graph_dir.iterdir()):
            if not community_dir.is_dir():
                continue
            
            community_name = community_dir.name
            
            # Get all available months for this community (sorted)
            available_months = sorted([
                f.stem for f in community_dir.glob('*.pt')
            ])
            
            # Skip if not enough data
            if len(available_months) < self.min_graphs_required:
                continue
            
            # Check each potential end month (t)
            for i, month_t in enumerate(available_months):
                # Skip if outside split range
                if not (start_month <= month_t <= end_month):
                    continue
                
                # Check if we have enough history
                if i < self.sequence_length - 1:
                    continue
                
                # Check if we have target at t+6
                target_idx = i + self.prediction_horizon
                if target_idx >= len(available_months):
                    continue
                
                # Get sequence and target months
                sequence_start_idx = i - self.sequence_length + 1
                sequence_months = available_months[sequence_start_idx : i + 1]
                target_month = available_months[target_idx]
                
                # Verify sequence is consecutive
                if self._is_consecutive_sequence(sequence_months, target_month):
                    samples.append({
                        'community': community_name,
                        'sequence_months': sequence_months,
                        'target_month': target_month,
                        'end_month': month_t
                    })
        
        return samples
    
    def _is_consecutive_sequence(
        self, 
        sequence_months: List[str], 
        target_month: str
    ) -> bool:
        """
        Verify that months form a consecutive sequence and target is exactly
        prediction_horizon months after the sequence end.
        
        Args:
            sequence_months: List of month strings in 'YYYY-MM' format
            target_month: Target month string in 'YYYY-MM' format
            
        Returns:
            True if sequence is consecutive and target is correctly positioned
        """
        try:
            # Verify we have correct sequence length
            if len(sequence_months) != self.sequence_length:
                return False
            
            # Check that sequence months are consecutive
            for i in range(len(sequence_months) - 1):
                curr = datetime.strptime(sequence_months[i], '%Y-%m')
                next_month = datetime.strptime(sequence_months[i + 1], '%Y-%m')
                expected_next = curr + relativedelta(months=1)
                
                if next_month != expected_next:
                    return False
            
            # Check that target is exactly prediction_horizon months after sequence end
            last_month = datetime.strptime(sequence_months[-1], '%Y-%m')
            target = datetime.strptime(target_month, '%Y-%m')
            expected_target = last_month + relativedelta(months=self.prediction_horizon)
            
            return target == expected_target
            
        except (ValueError, IndexError):
            return False
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading the graphs."""
        return self.samples[idx].copy()
    
    def get_communities(self) -> List[str]:
        """Get list of unique communities in this split."""
        return sorted(set(s['community'] for s in self.samples))
    
    def get_samples_by_community(self, community: str) -> List[int]:
        """Get indices of all samples for a specific community."""
        return [i for i, s in enumerate(self.samples) if s['community'] == community]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[List, Dict[str, float]]:
        """
        Load a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            graphs: List of 12 HeteroData graphs (input sequence)
            targets: Dict with keys 'qpd', 'answer_rate', 'retention', 'growth'
        """
        sample = self.samples[idx]
        community_dir = self.graph_dir / sample['community']
        
        # Load 12-month input sequence
        graphs = []
        for month in sample['sequence_months']:
            graph_path = community_dir / f"{month}.pt"
            try:
                graph = torch.load(graph_path, weights_only=False)
                graphs.append(graph)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading graph for {sample['community']} at {month}: {e}"
                )
        
        # Load target from t+6
        target_path = community_dir / f"{sample['target_month']}.pt"
        try:
            target_graph = torch.load(target_path, weights_only=False)
            targets = target_graph.y
        except Exception as e:
            raise RuntimeError(
                f"Error loading target for {sample['community']} at {sample['target_month']}: {e}"
            )
        
        # Verify targets exist and have correct format
        if targets is None or not isinstance(targets, dict):
            raise ValueError(
                f"Invalid targets for {sample['community']} at {sample['target_month']}: {targets}"
            )
        
        required_keys = {'qpd', 'answer_rate', 'retention', 'growth'}
        if not required_keys.issubset(targets.keys()):
            raise ValueError(
                f"Missing target keys for {sample['community']} at {sample['target_month']}. "
                f"Expected {required_keys}, got {set(targets.keys())}"
            )
        
        return graphs, targets


def collate_temporal_batch(batch: List[Tuple]) -> Tuple[List[List], Dict[str, torch.Tensor]]:
    """
    Collate function for batching temporal graph sequences.
    
    Since graphs have variable sizes (different numbers of users and tags),
    we cannot stack them into a single tensor. Instead, we keep them as
    a list of lists and let the model handle them individually.
    
    Args:
        batch: List of (graphs, targets) tuples from dataset
        
    Returns:
        batch_graphs: List of lists [batch_size][12 graphs]
        batch_targets: Dict of tensors for each metric [batch_size]
    """
    batch_graphs = []
    batch_targets = {
        'qpd': [],
        'answer_rate': [],
        'retention': [],
        'growth': []
    }
    
    for graphs, targets in batch:
        batch_graphs.append(graphs)
        for key in batch_targets.keys():
            batch_targets[key].append(targets[key])
    
    # Stack targets into tensors
    for key in batch_targets.keys():
        batch_targets[key] = torch.tensor(batch_targets[key], dtype=torch.float32)
    
    return batch_graphs, batch_targets


def create_dataloaders(
    graph_dir: str,
    batch_size: int = 8,
    val_batch_size: int = 16,
    test_batch_size: int = 16,
    num_workers: int = 4,
    sequence_length: int = 12,
    prediction_horizon: int = 6
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Convenience function to create train, val, and test dataloaders.
    
    Args:
        graph_dir: Root directory containing community graph folders
        batch_size: Batch size for training
        val_batch_size: Batch size for validation
        test_batch_size: Batch size for testing
        num_workers: Number of worker processes for data loading
        sequence_length: Number of months in input sequence
        prediction_horizon: Months ahead to predict
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TemporalCommunityDataset(
        graph_dir=graph_dir,
        split='train',
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    val_dataset = TemporalCommunityDataset(
        graph_dir=graph_dir,
        split='val',
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    test_dataset = TemporalCommunityDataset(
        graph_dir=graph_dir,
        split='test',
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_temporal_batch,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python temporal_dataset.py <graph_dir>")
        print("Example: python temporal_dataset.py data/processed/graphs")
        sys.exit(1)
    
    graph_dir = sys.argv[1]
    
    print("\n" + "="*70)
    print("TEMPORAL COMMUNITY DATASET - TEST")
    print("="*70)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TemporalCommunityDataset(graph_dir, split='train')
    val_dataset = TemporalCommunityDataset(graph_dir, split='val')
    test_dataset = TemporalCommunityDataset(graph_dir, split='test')
    
    # Print statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"  Train: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} ({len(val_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} ({len(test_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
    
    # Test loading a sample
    if len(train_dataset) > 0:
        print(f"\n{'='*70}")
        print("SAMPLE TEST")
        print("="*70)
        
        graphs, targets = train_dataset[0]
        sample_info = train_dataset.get_sample_info(0)
        
        print(f"\nSample 0 info:")
        print(f"  Community: {sample_info['community']}")
        print(f"  Sequence: {sample_info['sequence_months'][0]} to {sample_info['sequence_months'][-1]}")
        print(f"  Target month: {sample_info['target_month']}")
        
        print(f"\nLoaded data:")
        print(f"  Number of graphs: {len(graphs)}")
        print(f"  First graph: {graphs[0]}")
        print(f"    Users: {graphs[0]['user'].x.shape[0]}")
        print(f"    Tags: {graphs[0]['tag'].x.shape[0]}")
        print(f"  Targets: {targets}")
        
        # Test collate function
        print(f"\n{'='*70}")
        print("BATCH TEST")
        print("="*70)
        
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        batch_graphs, batch_targets = collate_temporal_batch(batch)
        
        print(f"\nBatch size: {len(batch_graphs)}")
        print(f"Batch targets shape:")
        for key, value in batch_targets.items():
            print(f"  {key}: {value.shape}")
        print(f"  Values: {batch_targets}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70 + "\n")