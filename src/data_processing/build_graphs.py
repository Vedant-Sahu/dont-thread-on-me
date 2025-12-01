"""
Build hierarchical PyG graphs for all Stack Exchange sites and all months.

Saves graphs as .pt files for efficient loading during training.

Usage:
    # Build graphs for all sites
    python src/data_processing/build_graphs.py --data_dir ~/Desktop/CS224W/parsed --output_dir ~/Desktop/CS224W/graphs
    
    # Build graphs for specific sites
    python src/data_processing/build_graphs.py --data_dir ~/Desktop/CS224W/parsed --output_dir ~/Desktop/CS224W/graphs --sites arduino.stackexchange.com astronomy.stackexchange.com
"""

import argparse
import itertools
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# defined in feature_extraction.py
from src.data_processing.feature_extraction import (
    build_tag_features,
    build_user_features,
    compute_community_metrics,
)


def _group_comments_by_month(comments: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group comments by month based on creation_date.
    
    Args:
        comments: List of comment dictionaries with 'creation_date' field
        
    Returns:
        Dictionary mapping month (YYYY-MM) to list of comments
    """
    monthly_comments = defaultdict(list)
    
    for comment in comments:
        creation_date = comment.get('creation_date')
        if not creation_date:
            continue
        
        # Extract YYYY-MM from creation_date
        try:
            month = creation_date[:7]  # 'YYYY-MM'
            monthly_comments[month].append(comment)
        except (IndexError, TypeError):
            continue
    
    return dict(monthly_comments)


def load_site_data(site_folder: Path) -> Optional[Dict]:
    """Load all data files for a site."""
    posts_path = site_folder / "monthly_posts.pkl.gz"
    users_path = site_folder / "users.pkl.gz"
    tags_path = site_folder / "tags.pkl.gz"
    comments_path = site_folder / "comments.pkl.gz"
    
    if not posts_path.exists():
        return None
    
    try:
        # Load data files (posts already grouped by month, comments need grouping)
        posts = pd.read_pickle(posts_path, compression="gzip")
        users = pd.read_pickle(users_path, compression="gzip") if users_path.exists() else {}
        tags = pd.read_pickle(tags_path, compression="gzip") if tags_path.exists() else {}
        
        # Comments come as flat list, group by month to match posts structure
        comments_list = pd.read_pickle(comments_path, compression="gzip") if comments_path.exists() else []
        comments = _group_comments_by_month(comments_list) if comments_list else {}
        
        return {
            'posts': posts,
            'users': users,
            'tags': tags,
            'comments': comments,
        }
    except Exception as e:
        print(f"Error loading {site_folder.name}: {e}")
        return None


def create_hetero_graph_with_features(
    posts: Dict,
    users: Dict,
    comments: Dict,
    month: str,
    prev_month: Optional[str] = None,
    next_month: Optional[str] = None
) -> Optional[HeteroData]:
    """
    Create PyG HeteroData with:
    - tag-tag co-occurrence edges
    - user-tag bipartite edges
    - Node features for both tags and users
    """
    data = HeteroData()
    
    if month not in posts:
        return None
    
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    # Skip if no activity
    if not questions and not answers:
        return None
    
    # ===== Collect all tags and users =====
    tag_set = set()
    user_set = set()
    
    for q in questions:
        tag_set.update(q['tags'])
        if q['user_id']:
            user_set.add(q['user_id'])
    
    for a in answers:
        tag_set.update(a['parent_tags'])
        if a['user_id']:
            user_set.add(a['user_id'])
    
    if not tag_set or not user_set:
        return None
    
    # Create index mappings
    tag_to_idx = {tag: i for i, tag in enumerate(sorted(tag_set))}
    user_to_idx = {user: i for i, user in enumerate(sorted(user_set))}
    
    # ===== Level 1: Tag-Tag Co-occurrence =====
    tag_cooccurrence = defaultdict(int)
    
    for q in questions:
        tags = q['tags']
        for tag1, tag2 in itertools.combinations(sorted(tags), 2):
            tag_cooccurrence[(tag1, tag2)] += 1
    
    tag_edges = []
    tag_weights = []
    
    for (tag1, tag2), weight in tag_cooccurrence.items():
        idx1, idx2 = tag_to_idx[tag1], tag_to_idx[tag2]
        tag_edges.append([idx1, idx2])
        tag_edges.append([idx2, idx1])
        tag_weights.extend([weight, weight])
    
    if tag_edges:
        data['tag', 'cooccurs', 'tag'].edge_index = torch.tensor(tag_edges, dtype=torch.long).t()
        data['tag', 'cooccurs', 'tag'].edge_weight = torch.tensor(tag_weights, dtype=torch.float)
    
    # ===== Level 2: User-Tag Bipartite =====
    user_tag_edges = defaultdict(int)
    
    for q in questions:
        if q['user_id']:
            user_idx = user_to_idx[q['user_id']]
            for tag in q['tags']:
                tag_idx = tag_to_idx[tag]
                user_tag_edges[(user_idx, tag_idx)] += 1
    
    for a in answers:
        if a['user_id']:
            user_idx = user_to_idx[a['user_id']]
            for tag in a['parent_tags']:
                tag_idx = tag_to_idx[tag]
                user_tag_edges[(user_idx, tag_idx)] += 1
    
    ut_edges = []
    ut_weights = []
    
    for (user_idx, tag_idx), weight in user_tag_edges.items():
        ut_edges.append([user_idx, tag_idx])
        ut_weights.append(weight)
    
    if ut_edges:
        edge_index_tensor = torch.tensor(ut_edges, dtype=torch.long).t()
        edge_weight_tensor = torch.tensor(ut_weights, dtype=torch.float)
        
        # Forward direction
        data['user', 'contributes', 'tag'].edge_index = edge_index_tensor
        data['user', 'contributes', 'tag'].edge_weight = edge_weight_tensor
        
        # Reverse direction (flip source/target)
        data['tag', 'contributed_to_by', 'user'].edge_index = edge_index_tensor.flip(0)
        data['tag', 'contributed_to_by', 'user'].edge_weight = edge_weight_tensor
    
    # ===== Build Tag Features =====
    tag_features_dict = build_tag_features(posts, comments, month, prev_month, tag_set)
    
    # Extract features in sorted order
    tag_feature_matrix = []
    for tag in sorted(tag_set):
        feats = tag_features_dict[tag]
        feature_vector = [
            feats['post_popularity'],
            feats['comment_popularity'],
            feats['avg_views'],
            feats['answer_quality'],
            feats['difficulty'],
            feats['diversity'],
            feats['growth_rate']
        ]
        tag_feature_matrix.append(feature_vector)
    
    data['tag'].x = torch.tensor(tag_feature_matrix, dtype=torch.float)
    
    # ===== Build User Features =====
    user_features_dict = build_user_features(posts, comments, users, month, next_month)
    
    user_feature_matrix = []
    for user_id in sorted(user_set):
        feats = user_features_dict[user_id]
        feature_vector = [
            feats['reputation'],
            feats['tenure'],
            feats['activity'],
            feats['expertise_entropy'],
            feats['retention']
        ]
        user_feature_matrix.append(feature_vector)
    
    data['user'].x = torch.tensor(user_feature_matrix, dtype=torch.float)
    
    # ===== Compute and Store Community-level Target Metrics =====
    community_metrics = compute_community_metrics(posts, users, month, prev_month)
    if community_metrics:
        data.y = community_metrics
    
    # Store metadata
    data['tag'].tag_to_idx = tag_to_idx
    data['user'].user_to_idx = user_to_idx
    
    return data


def get_sorted_months(posts: Dict) -> List[str]:
    """Get chronologically sorted list of months."""
    months = [m for m in posts.keys() if m and m != 'metadata']
    return sorted(months)


def build_graphs_for_site(
    site_folder: Path,
    output_folder: Path,
    skip_existing: bool = True
) -> Dict:
    """Build all monthly graphs for a single site."""
    site_name = site_folder.name
    print(f"\n{'='*60}")
    print(f"Processing: {site_name}")
    print(f"{'='*60}")
    
    # Load data
    data = load_site_data(site_folder)
    if data is None:
        print(f"Skipping {site_name}: Could not load data")
        return {'built': 0, 'skipped': 0, 'errors': 1}
    
    posts = data['posts']
    users = data['users']
    comments = data.get('comments', {})
    
    # Get all months
    months = get_sorted_months(posts)
    if not months:
        print(f"Skipping {site_name}: No months found")
        return {'built': 0, 'skipped': 0, 'errors': 1}
    
    print(f"Found {len(months)} months: {months[0]} to {months[-1]}")
    
    # Create output directory for this site
    site_output = output_folder / site_name
    site_output.mkdir(parents=True, exist_ok=True)
    
    # Build graph for each month
    built_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, month in enumerate(tqdm(months, desc=f"  {site_name}", leave=False)):
        output_path = site_output / f"{month}.pt"
        
        # Skip if already exists
        if skip_existing and output_path.exists():
            skipped_count += 1
            continue
        
        # Get adjacent months for temporal features
        prev_month = months[i-1] if i > 0 else None
        next_month = months[i+1] if i < len(months)-1 else None
        
        # Build graph
        try:
            graph = create_hetero_graph_with_features(posts, users, comments, month, prev_month, next_month)
            
            if graph is not None:
                torch.save(graph, output_path)
                built_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            import traceback
            print(f"\nError processing {site_name} {month}: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
            error_count += 1
            skipped_count += 1
    
    print(f"  Built: {built_count} graphs, Skipped: {skipped_count}, Errors: {error_count}")
    return {'built': built_count, 'skipped': skipped_count, 'errors': error_count}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build PyG graphs for Stack Exchange sites')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing parsed site data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save built graphs')
    parser.add_argument('--sites', type=str, nargs='+', default=None, help='Specific sites to process')
    parser.add_argument('--no-skip-existing', action='store_true', help='Rebuild graphs even if they already exist')
    return parser.parse_args()


def get_site_folders(data_dir: Path, sites: Optional[List[str]]) -> List[Path]:
    """Get list of site folders to process."""
    if sites:
        return [data_dir / site for site in sites]
    return [f for f in data_dir.iterdir() if f.is_dir()]


def process_all_sites(site_folders: List[Path], output_dir: Path, skip_existing: bool) -> Dict:
    """Process all sites and return aggregate statistics."""
    total_stats = {'built': 0, 'skipped': 0, 'errors': 0}
    for site_folder in site_folders:
        stats = build_graphs_for_site(site_folder, output_dir, skip_existing=skip_existing)
        total_stats['built'] += stats['built']
        total_stats['skipped'] += stats['skipped']
        total_stats['errors'] += stats['errors']
    return total_stats


def print_summary(total_stats: Dict, output_dir: Path):
    """Print final summary."""
    print(f"\nTotal built: {total_stats['built']}, skipped: {total_stats['skipped']}, errors: {total_stats['errors']}")
    print(f"Graphs saved to: {output_dir}")


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        sys.exit(f"Error: Data directory does not exist: {data_dir}")
    
    site_folders = get_site_folders(data_dir, args.sites)
    if not site_folders:
        sys.exit("Error: No sites found to process")
    
    total_stats = process_all_sites(site_folders, output_dir, skip_existing=not args.no_skip_existing)
    print_summary(total_stats, output_dir)


if __name__ == "__main__":
    main()