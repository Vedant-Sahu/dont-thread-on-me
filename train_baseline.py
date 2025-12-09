#!/usr/bin/env python3
"""Train linear extrapolation baseline on full dataset."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.baselines import create_baseline, evaluate_baseline
from src.data_processing.temporal_dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train linear extrapolation baseline")
    parser.add_argument(
        '--graphs_dir',
        type=str,
        required=True,
        help='Path to directory containing community graph folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baseline',
        help='Directory to save model and results (default: results/baseline)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    args = parser.parse_args()
    
    graphs_dir = Path(args.graphs_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not graphs_dir.exists():
        print(f"Error: Graphs directory does not exist: {graphs_dir}")
        return
    
    print("="*70)
    print("LINEAR EXTRAPOLATION BASELINE TRAINING")
    print("="*70)
    print(f"Graphs directory: {graphs_dir}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        graph_dir=str(graphs_dir),
        batch_size=args.batch_size,
        val_batch_size=64,
        test_batch_size=64,
        num_workers=4
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create and train baseline
    print("\nTraining baseline...")
    baseline = create_baseline()
    baseline.fit(train_loader, show_progress=True)
    print("  ✓ Training complete")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_baseline(baseline, val_loader, show_progress=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_baseline(baseline, test_loader, show_progress=True)
    
    # Save model
    model_path = output_dir / "baseline_model.pkl"
    baseline.save(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'val_results': val_results,
        'test_results': test_results,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset)
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nValidation Set:")
    for metric in ['qpd', 'answer_rate', 'retention', 'growth']:
        if metric in val_results:
            print(f"  {metric:15s} | MSE: {val_results[metric]['mse']:8.4f} | "
                  f"MAE: {val_results[metric]['mae']:8.4f} | "
                  f"R²: {val_results[metric]['r2']:8.4f}")
    
    if 'overall' in val_results:
        print(f"\n  Overall R²: {val_results['overall']['r2']:.4f}")
    
    print("\nTest Set:")
    for metric in ['qpd', 'answer_rate', 'retention', 'growth']:
        if metric in test_results:
            print(f"  {metric:15s} | MSE: {test_results[metric]['mse']:8.4f} | "
                  f"MAE: {test_results[metric]['mae']:8.4f} | "
                  f"R²: {test_results[metric]['r2']:8.4f}")
    
    if 'overall' in test_results:
        print(f"\n  Overall R²: {test_results['overall']['r2']:.4f}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()

