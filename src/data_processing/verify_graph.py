"""
Quick verification script to check if graph objects were created and saved correctly.

Usage:
    python src/data_processing/verify_graph.py --sites arduino.stackexchange.com astronomy.stackexchange.com
"""

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



def verify_graph(graphs_dir: Path, site_name: str, month: str) -> bool:
    """Verify a single graph file."""
    graph_path = graphs_dir / site_name / f"{month}.pt"
    
    if not graph_path.exists():
        print(f"{month}: File not found")
        return False
    
    try:
        graph = torch.load(graph_path, weights_only=False)
        
        if not isinstance(graph, HeteroData):
            print(f"{month}: Not a HeteroData object (got {type(graph)})")
            return False
        
        # Check required components
        checks = []
        
        # Check tag node features
        if 'tag' not in graph.node_types:
            checks.append("Missing 'tag' node type")
        elif 'x' not in graph['tag']:
            checks.append("Missing tag features (x)")
        else:
            tag_shape = graph['tag'].x.shape
            checks.append(f"Tag features: {tag_shape[0]} tags, {tag_shape[1]} features")
        
        # Check user node features
        if 'user' not in graph.node_types:
            checks.append("Missing 'user' node type")
        elif 'x' not in graph['user']:
            checks.append("Missing user features (x)")
        else:
            user_shape = graph['user'].x.shape
            checks.append(f"User features: {user_shape[0]} users, {user_shape[1]} features")
        
        # Check tag-tag edges
        if ('tag', 'cooccurs', 'tag') not in graph.edge_types:
            checks.append("Missing tag-tag edges")
        else:
            edge_count = graph['tag', 'cooccurs', 'tag'].edge_index.shape[1]
            checks.append(f"Tag-tag edges: {edge_count}")
        
        # Check user-tag edges
        if ('user', 'contributes', 'tag') not in graph.edge_types:
            checks.append("Missing user-tag edges")
        else:
            edge_count = graph['user', 'contributes', 'tag'].edge_index.shape[1]
            checks.append(f"User-tag edges: {edge_count}")
        
        # Check mappings
        if 'tag_to_idx' not in graph['tag']:
            checks.append("Missing tag_to_idx mapping")
        else:
            checks.append(f"Tag mappings: {len(graph['tag'].tag_to_idx)} tags")
        
        if 'user_to_idx' not in graph['user']:
            checks.append("Missing user_to_idx mapping")
        else:
            checks.append(f"User mappings: {len(graph['user'].user_to_idx)} users")
        
        print(f"{month}: Graph loaded successfully")
        for check in checks:
            print(f"      {check}")
        
        return True
        
    except Exception as e:
        print(f"{month}: Error loading graph - {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify graph objects were created correctly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify specific sites
  python src/data_processing/verify_graph.py --sites arduino.stackexchange.com astronomy.stackexchange.com
  
  # Verify all months for a site
  python src/data_processing/verify_graph.py --sites arduino.stackexchange.com --all-months
        """
    )
    
    parser.add_argument(
        "--sites",
        nargs="+",
        required=True,
        help="Site names to verify (e.g., 'arduino.stackexchange.com')"
    )
    
    parser.add_argument(
        "--all-months",
        action="store_true",
        help="Verify all months for each site (default: verify first month only)"
    )
    
    parser.add_argument(
        "--months",
        nargs="+",
        help="Specific months to verify (e.g., '2010-08 2010-09')"
    )
    
    parser.add_argument(
        "--graphs_dir",
        type=str,
        required=True,
        help="Directory containing saved graphs"
    )
    
    args = parser.parse_args()
    
    graphs_dir = Path(args.graphs_dir).expanduser()
    
    print("=" * 60)
    print("Graph Verification")
    print("=" * 60)
    print(f"Graphs directory: {graphs_dir}")
    print()
    
    all_valid = True
    
    for site_name in args.sites:
        print(f"\nSite: {site_name}")
        print("-" * 60)
        
        site_dir = graphs_dir / site_name
        
        if not site_dir.exists():
            print(f"Site directory does not exist: {site_dir}")
            all_valid = False
            continue
        
        # Get all graph files
        graph_files = sorted(site_dir.glob("*.pt"))
        
        if not graph_files:
            print(f" No graph files found in {site_dir}")
            all_valid = False
            continue
        
        print(f"  Found {len(graph_files)} graph file(s)")
        
        # Determine which months to verify
        if args.months:
            months_to_check = args.months
        elif args.all_months:
            months_to_check = [f.stem for f in graph_files]
        else:
            # Just check the first one
            months_to_check = [graph_files[0].stem]
        
        # Verify each month
        for month in months_to_check:
            is_valid = verify_graph(graphs_dir, site_name, month)
            if not is_valid:
                all_valid = False
    
    print()
    print("=" * 60)
    if all_valid:
        print("All graphs verified successfully!")
    else:
        print("Some graphs failed verification")
    print("=" * 60)


if __name__ == "__main__":
    main()

