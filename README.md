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
