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
