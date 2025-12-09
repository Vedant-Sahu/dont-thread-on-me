# Don't Thread on Me

Predicting online community health trajectories using temporal graph neural networks.

## Collaborators
- David Lupea
- Kalyani Limaye
- Vedant Sahu

## Project Overview
This project implements a temporal GNN framework to predict Stack Exchange community health metrics and recommend optimal structural interventions (split, merge, scope changes).

## Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: Source code for models and processing
- `notebooks/`: Jupyter notebooks for exploration
- `experiments/`: Training scripts and configs
- `results/`: Model outputs and visualizations
- `docs/`: Project documentation

## Usage

### Building Graphs

Build PyG graphs from parsed Stack Exchange data:

```bash
# Build graphs for all sites
python src/data_processing/build_graphs.py --data_dir /path/to/parsed --output_dir /path/to/graphs

# Build graphs for specific sites
python src/data_processing/build_graphs.py --data_dir /path/to/parsed --output_dir /path/to/graphs --sites arduino.stackexchange.com astronomy.stackexchange.com

# Force rebuild (don't skip existing graphs)
python src/data_processing/build_graphs.py --data_dir /path/to/parsed --output_dir /path/to/graphs --no-skip-existing
```

### Verifying Graphs

Verify that graph objects were created correctly:

```bash
# Verify graphs for specific sites
python src/data_processing/verify_graph.py --graphs_dir /path/to/graphs --sites arduino.stackexchange.com astronomy.stackexchange.com

# Verify all months for a site
python src/data_processing/verify_graph.py --graphs_dir /path/to/graphs --sites arduino.stackexchange.com --all-months
```

**Note:** Replace `/path/to/parsed` and `/path/to/graphs` with your local data directories. Each team member should use their own paths via CLI arguments.

### Visualizations

The `notebooks/visualizations_for_report.ipynb` notebook contains visualization functions for analyzing graph data and community health metrics without requiring full model training. It includes:

- **Temporal Evolution**: Community health metrics (QPD, answer rate, retention, growth) over time
- **Tag Popularity Trajectories**: Top N tags and their popularity trends
- **Feature Correlations**: Heatmaps showing relationships between tag and user features
- **Graph Structure Statistics**: Evolution of nodes, edges, and degrees over time
- **Feature-Target Correlations**: Which features are most predictive of community health

To use, update the `GRAPHS_DIR` and `SITES` variables at the top of the notebook and run the cells.

### Baseline Model

A linear extrapolation baseline is provided for comparison with the temporal GNN. The baseline:

- Extracts **23 temporal and graph-aggregated features** from 12-month graph sequences:
  - **Trends** (4 features): Linear slopes of community metrics over last 6 months
  - **Means** (4 features): Average values of community metrics over 12 months
  - **Graph features** (15 features): Aggregated tag features, user features, and graph size statistics
- Trains **4 separate linear regression models** (one per health metric: qpd, answer_rate, retention, growth)
- Uses standardized features and predicts metrics **6 months ahead**

**Training the Baseline:**

```bash
# Train on full dataset
python train_baseline.py --graphs_dir /path/to/graphs --output_dir results/baseline

# Example
python train_baseline.py --graphs_dir ~/Desktop/CS224W/graphs --output_dir results/baseline
```

**Saved Outputs:**

After training, the following files are saved to the output directory (default: `results/baseline/`):

1. **`baseline_model.pkl`**: Trained model file containing:
   - 4 fitted `LinearRegression` models (one per metric)
   - 4 fitted `StandardScaler` objects (one per metric)
   - Model state flags

2. **`results.json`**: Evaluation results containing:
   - Validation set metrics (MSE, MAE, R², RMSE) for each metric
   - Test set metrics (MSE, MAE, R², RMSE) for each metric
   - Overall aggregated metrics
   - Sample counts for train/val/test splits
   - Timestamp of training run

**Using a Saved Model:**

```python
from pathlib import Path
from src.baselines import LinearExtrapolationBaseline
from torch_geometric.data import HeteroData

# Load trained model
model_path = Path("results/baseline/baseline_model.pkl")
baseline = LinearExtrapolationBaseline.load(model_path)

# Predict on a single 12-month sequence
# monthly_graphs is a list of 12 HeteroData objects
predictions = baseline.predict_single(monthly_graphs)
# Returns: {'qpd': float, 'answer_rate': float, 'retention': float, 'growth': float}

# Predict on a batch of sequences
# batch_graphs is a list of lists, each inner list has 12 graphs
batch_predictions = baseline.predict(batch_graphs)
# Returns: {'qpd': np.ndarray, 'answer_rate': np.ndarray, ...}
```

**Note:** The saved model can be shared with team members and used for inference without retraining. Ensure the model is loaded on a system with the same Python dependencies (scikit-learn, numpy, torch-geometric).
