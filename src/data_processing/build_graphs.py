"""
Build hierarchical PyG graphs for all Stack Exchange sites and all months.

Saves graphs as .pt files for efficient loading during training.

Usage:
    # Build graphs for all sites
    python src/data_processing/build_graphs.py --data_dir data/processed/parsed --output_dir data/processed/graphs
    
    # Build graphs for specific sites
    python src/data_processing/build_graphs.py --data_dir data/processed/parsed --output_dir data/processed/graphs --sites arduino.stackexchange.com astronomy.stackexchange.com
"""

import argparse
import itertools
import os
import pickle
import sys
import math
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from scipy.stats import entropy
from calendar import monthrange

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    Optimized implementation - O(posts + answers + comments)
    
    1. Single pass through all data to build all aggregations
    2. Hash map lookups instead of repeated iterations
    3. Pre-compute answer quality per tag in one pass through all months
    4. Pre-compute all user features without per-user iteration
    """

    data = HeteroData()
    
    if month not in posts:
        return None
    
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    if not questions and not answers:
        return None
    
    # ========== PHASE 1: Single pass through current month data ==========
    # Build all data structures in ONE pass
    
    tag_set = set()
    user_set = set()
    tag_cooccurrence = defaultdict(int)
    user_tag_edges_raw = defaultdict(int)  # (user_id, tag) -> count
    
    # Tag statistics (for features)
    tag_question_count = defaultdict(int)
    tag_view_sum = defaultdict(float)
    tag_view_count = defaultdict(int)
    tag_no_accepted_count = defaultdict(int)  # For difficulty
    tag_user_counts = defaultdict(lambda: defaultdict(int))  # tag -> user -> count (for diversity)
    
    # For comment counting
    tag_post_ids = defaultdict(set)  # tag -> set of post_ids (questions + answers)
    question_tags = {}  # question_id -> tags (for answer lookup)
    
    # User statistics (for features)
    user_question_count = defaultdict(int)
    user_answer_count = defaultdict(int)
    user_tag_counts = defaultdict(lambda: defaultdict(int))  # user -> tag -> count
    
    # Single pass through questions
    for q in questions:
        q_tags = q['tags']
        q_user_id = q.get('user_id')
        q_post_id = q.get('post_id')
        q_views = q.get('view_count', 0)
        has_accepted = q.get('accepted_answer_id') is not None
        
        tag_set.update(q_tags)
        question_tags[q_post_id] = q_tags
        
        if q_user_id:
            user_set.add(q_user_id)
            user_question_count[q_user_id] += 1
        
        # Tag statistics 
        for tag in q_tags:
            tag_question_count[tag] += 1
            tag_view_sum[tag] += q_views
            if q_views > 0:
                tag_view_count[tag] += 1
            if not has_accepted:
                tag_no_accepted_count[tag] += 1
            if q_post_id:
                tag_post_ids[tag].add(q_post_id)
            if q_user_id:
                tag_user_counts[tag][q_user_id] += 1
                user_tag_edges_raw[(q_user_id, tag)] += 1
                user_tag_counts[q_user_id][tag] += 1
        
        # Tag co-occurrence
        if len(q_tags) > 1:
            sorted_q_tags = sorted(q_tags)
            for i in range(len(sorted_q_tags)):
                for j in range(i + 1, len(sorted_q_tags)):
                    tag_cooccurrence[(sorted_q_tags[i], sorted_q_tags[j])] += 1
    
    # Single pass through answers
    for a in answers:
        a_tags = a.get('parent_tags', [])
        a_user_id = a.get('user_id')
        a_post_id = a.get('post_id')
        
        tag_set.update(a_tags)
        
        if a_user_id:
            user_set.add(a_user_id)
            user_answer_count[a_user_id] += 1
        
        for tag in a_tags:
            if a_post_id:
                tag_post_ids[tag].add(a_post_id)
            if a_user_id:
                tag_user_counts[tag][a_user_id] += 1
                user_tag_edges_raw[(a_user_id, tag)] += 1
                user_tag_counts[a_user_id][tag] += 1
    
    if not tag_set or not user_set:
        return None
    
    # ========== PHASE 2: Pre-compute answer quality for all tags in one pass ==========
    # Iterate through all months once and aggregate per tag
    
    tag_answer_scores = defaultdict(list)  # tag -> list of scores
    
    # Get question IDs for this month
    month_question_ids = set(question_tags.keys())
    
    # Single pass through ALL months to find answers to this month's questions
    for answer_month in posts.keys():
        if answer_month == 'metadata':
            continue
        month_answers = posts[answer_month].get('answers', [])
        for a in month_answers:
            parent_id = a.get('parent_id')
            parent_month = a.get('parent_month')
            # Only count answers to questions from our target month
            if parent_id in month_question_ids and parent_month == month:
                score = a.get('score', 0)
                # Get tags from the question
                if parent_id in question_tags:
                    for tag in question_tags[parent_id]:
                        tag_answer_scores[tag].append(score)
    
    # ========== PHASE 3: Pre-compute previous month tag counts ==========
    prev_tag_question_count = defaultdict(int)
    if prev_month and prev_month in posts:
        for q in posts[prev_month]['questions']:
            for tag in q.get('tags', []):
                prev_tag_question_count[tag] += 1
    
    # ========== PHASE 4: Pre-compute comment counts (single pass) ==========
    month_comments = comments.get(month, [])
    post_comment_counts = defaultdict(int)
    user_comment_count = defaultdict(int)
    
    for c in month_comments:
        post_id = c.get('post_id')
        c_user_id = c.get('user_id')
        if post_id:
            post_comment_counts[post_id] += 1
        if c_user_id:
            user_comment_count[c_user_id] += 1
    
    # ========== PHASE 5: Pre-compute next month active users ==========
    next_month_active_users = set()
    if next_month and next_month in posts:
        for q in posts[next_month]['questions']:
            if q.get('user_id'):
                next_month_active_users.add(q['user_id'])
        for a in posts[next_month]['answers']:
            if a.get('user_id'):
                next_month_active_users.add(a['user_id'])
        for c in comments.get(next_month, []):
            if c.get('user_id'):
                next_month_active_users.add(c['user_id'])
    
    # ========== PHASE 6: Build graph structure ==========
    sorted_tags = sorted(tag_set)
    sorted_users = sorted(user_set)
    tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
    user_to_idx = {user: i for i, user in enumerate(sorted_users)}
    
    # Tag-Tag edges
    if tag_cooccurrence:
        tag_edges_list = []
        tag_weights_list = []
        
        for (tag1, tag2), weight in tag_cooccurrence.items():
            idx1, idx2 = tag_to_idx[tag1], tag_to_idx[tag2]
            tag_edges_list.append([idx1, idx2])
            tag_edges_list.append([idx2, idx1])
            tag_weights_list.extend([weight, weight])
        
        data['tag', 'cooccurs', 'tag'].edge_index = torch.tensor(tag_edges_list, dtype=torch.long).t()
        data['tag', 'cooccurs', 'tag'].edge_weight = torch.tensor(tag_weights_list, dtype=torch.float)
    
    # User-Tag edges
    if user_tag_edges_raw:
        ut_edges_list = []
        ut_weights_list = []
        
        for (user_id, tag), weight in user_tag_edges_raw.items():
            user_idx = user_to_idx[user_id]
            tag_idx = tag_to_idx[tag]
            ut_edges_list.append([user_idx, tag_idx])
            ut_weights_list.append(weight)
        
        edge_index_tensor = torch.tensor(ut_edges_list, dtype=torch.long).t()
        edge_weight_tensor = torch.tensor(ut_weights_list, dtype=torch.float)
        
        data['user', 'contributes', 'tag'].edge_index = edge_index_tensor
        data['user', 'contributes', 'tag'].edge_weight = edge_weight_tensor
        data['tag', 'contributed_to_by', 'user'].edge_index = edge_index_tensor.flip(0)
        data['tag', 'contributed_to_by', 'user'].edge_weight = edge_weight_tensor
    
    # ========== PHASE 7: Build tag features from pre-computed data ==========
    tag_feature_matrix = []
    
    for tag in sorted_tags:
        q_count = tag_question_count[tag]
        
        # post_popularity
        post_pop = q_count
        
        # comment_popularity - use pre-computed post_comment_counts
        comment_pop = sum(post_comment_counts[pid] for pid in tag_post_ids[tag])
        
        # avg_views
        avg_views = tag_view_sum[tag] / q_count if q_count > 0 else 0.0
        
        # answer_quality - use pre-computed tag_answer_scores
        scores = tag_answer_scores[tag]
        if scores:
            k = max(1, math.ceil(0.1 * len(scores)))
            sorted_scores = sorted(scores, reverse=True)[:k]
            answer_quality = sum(sorted_scores) / len(sorted_scores)
        else:
            answer_quality = 0.0
        
        # difficulty - fraction without accepted answer
        difficulty = tag_no_accepted_count[tag] / q_count if q_count > 0 else 0.0
        
        # diversity - entropy of user distribution
        user_counts = list(tag_user_counts[tag].values())
        diversity = float(entropy(user_counts)) if user_counts else 0.0
        
        # growth_rate
        prev_pop = prev_tag_question_count[tag]
        if prev_pop == 0:
            growth_rate = 0.0 if q_count == 0 else 1.0
        else:
            growth_rate = (q_count - prev_pop) / prev_pop
        
        tag_feature_matrix.append([
            post_pop,
            comment_pop,
            avg_views,
            answer_quality,
            difficulty,
            diversity,
            growth_rate
        ])
    
    data['tag'].x = torch.tensor(tag_feature_matrix, dtype=torch.float)
    
    # ========== PHASE 8: Build user features from pre-computed data ==========
    # Pre-parse current month for tenure calculation
    current_month_dt = datetime.strptime(month, '%Y-%m')
    
    user_feature_matrix = []
    
    for user_id in sorted_users:
        # reputation
        reputation = users.get(user_id, {}).get('reputation', 0) if users else 0
        
        # tenure
        tenure = 0
        if users and user_id in users:
            creation_date = users[user_id].get('creation_date')
            if creation_date:
                try:
                    user_join = datetime.strptime(creation_date[:7], '%Y-%m')
                    months_diff = (current_month_dt.year - user_join.year) * 12 + (current_month_dt.month - user_join.month)
                    tenure = max(0, months_diff)
                except:
                    pass
        
        # activity - use pre-computed counts
        activity = user_question_count[user_id] + user_answer_count[user_id] + user_comment_count[user_id]
        
        # expertise_entropy - use pre-computed user_tag_counts
        tag_counts = list(user_tag_counts[user_id].values())
        expertise_entropy = float(entropy(tag_counts)) if tag_counts else 0.0
        
        # retention - use pre-computed next_month_active_users
        retention = 1 if user_id in next_month_active_users else 0
        
        user_feature_matrix.append([
            reputation,
            tenure,
            activity,
            expertise_entropy,
            retention
        ])
    
    data['user'].x = torch.tensor(user_feature_matrix, dtype=torch.float)
    
    # ========== PHASE 9: Community metrics (simplified) ==========
    # Compute inline instead of calling function
    year, month_num = map(int, month.split('-'))
    days_in_month = monthrange(year, month_num)[1]
    qpd = len(questions) / days_in_month
    
    questions_with_accepted = sum(1 for q in questions if q.get('accepted_answer_id') is not None)
    answer_rate = questions_with_accepted / len(questions) if questions else 0.0
    
    active_user_retentions = [1 if uid in next_month_active_users else 0 for uid in user_set]
    avg_retention = sum(active_user_retentions) / len(active_user_retentions) if active_user_retentions else 0.0
    
    current_month_dt = datetime.strptime(month, '%Y-%m')
    
    new_users_current = 0
    if users:
        for user_id in user_set:
            if user_id in users:
                creation_date = users[user_id].get('creation_date')
                if creation_date:
                    try:
                        user_join = datetime.strptime(creation_date[:7], '%Y-%m')
                        # User is "new" if they joined this month
                        if user_join.year == current_month_dt.year and user_join.month == current_month_dt.month:
                            new_users_current += 1
                    except (ValueError, AttributeError):
                        pass

    # Calculate new users for previous month
    growth = 0.0

    # Check prev_month is not None and exists in posts
    if prev_month is not None and prev_month in posts and users:
        prev_month_dt = datetime.strptime(prev_month, '%Y-%m')
    
    # Get active users in prev month  
    prev_user_set = set()
    for q in posts[prev_month].get('questions', []):
        if q.get('user_id'):
            prev_user_set.add(q['user_id'])
    for a in posts[prev_month].get('answers', []):
        if a.get('user_id'):
            prev_user_set.add(a['user_id'])    
    
    # Count new users who joined in previous month
    new_users_prev = 0
    for user_id in prev_user_set:
        if user_id in users:
            creation_date = users[user_id].get('creation_date')
            if creation_date:
                try:
                    user_join = datetime.strptime(creation_date[:7], '%Y-%m')
                    # User is "new" to prev month if they joined that month
                    if user_join.year == prev_month_dt.year and user_join.month == prev_month_dt.month:
                        new_users_prev += 1
                except (ValueError, AttributeError):
                    pass
    
    # Calculate growth rate: (current - prev) / prev
    if new_users_prev > 0:
        growth = (new_users_current - new_users_prev) / new_users_prev
    elif new_users_current > 0:
        growth = 1.0  # All growth if no new users previously

    data.y = {
        'qpd': float(qpd),
        'answer_rate': float(answer_rate),
        'retention': float(avg_retention),
        'growth': float(growth)
    }
    
    data['tag'].tag_to_idx = tag_to_idx
    data['user'].user_to_idx = user_to_idx
    
    return data


def get_sorted_months(posts: Dict) -> List[str]:
    """Get chronologically sorted list of months."""
    months = [m for m in posts.keys() if m and m != 'metadata']
    return sorted(months)


def check_existing_file_fast(file_path: Path) -> bool:
    """
    Fast check if a graph file exists and is likely valid.
    Uses file size check instead of loading (much faster).
    
    Returns:
        True if file exists and has non-zero size, False otherwise
    """
    try:
        return file_path.exists() and file_path.stat().st_size > 0
    except Exception:
        return False


def check_existing_file(output_path_str: str) -> bool:
    """
    Check if a graph file already exists. Used for parallel checking.
    
    Returns:
        True if file exists and is valid, False otherwise
    """
    output_path = Path(output_path_str)
    return check_existing_file_fast(output_path)


def process_single_task(
    site_folder_str: str,
    site_name: str,
    month: str,
    month_idx: int,
    all_months: List[str],
    output_path_str: str,
    skip_existing: bool
) -> Dict:
    """
    Process a single (site, month) task. This function loads site data and processes one month.
    Designed for multiprocessing across all sites and months.
    
    Returns:
        Dictionary with 'status' ('built', 'skipped', 'error'), 'site_name', 'month', and optional 'error' message
    """
    output_path = Path(output_path_str)
    site_folder = Path(site_folder_str)
    
    # Skip if already exists
    if skip_existing and output_path.exists():
        # Quick check - if file exists, assume it's valid (detailed check done in parallel scan)
        return {'status': 'skipped', 'site_name': site_name, 'month': month}
    
    # Load site data (each worker loads it independently)
    data = load_site_data(site_folder)
    if data is None:
        return {'status': 'error', 'site_name': site_name, 'month': month, 
                'error': f'Could not load data for {site_name}'}
    
    posts = data['posts']
    users = data['users']
    comments = data.get('comments', {})
    
    # Get adjacent months for temporal features
    prev_month = all_months[month_idx-1] if month_idx > 0 else None
    next_month = all_months[month_idx+1] if month_idx < len(all_months)-1 else None
    
    # Build graph
    try:
        graph = create_hetero_graph_with_features(posts, users, comments, month, prev_month, next_month)
        
        if graph is not None:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, output_path)
            return {'status': 'built', 'site_name': site_name, 'month': month}
        else:
            # Graph is None - likely no data for this month
            return {'status': 'skipped', 'site_name': site_name, 'month': month}
    except Exception as e:
        import traceback
        error_msg = f"Error processing {site_name} {month}: {e}\n{traceback.format_exc()}"
        return {'status': 'error', 'site_name': site_name, 'month': month, 'error': error_msg}


def process_site_months(
    site_folder_str: str,
    site_name: str,
    months_to_process: List[Tuple[str, int, List[str], str]],
    existing_files: Set[Tuple[str, str]]
) -> List[Dict]:
    """
    Process multiple months for a single site, loading the site data only once.
    This is much more efficient than loading the data for each month separately.
    
    Args:
        site_folder_str: Path to site folder
        site_name: Name of the site
        months_to_process: List of (month, month_idx, all_months, output_path_str) tuples
        existing_files: Set of (site_name, month) tuples that already exist
    
    Returns:
        List of result dictionaries
    """
    site_folder = Path(site_folder_str)
    
    # Load site data once
    data = load_site_data(site_folder)
    if data is None:
        return [{'status': 'error', 'site_name': site_name, 'month': month, 
                 'error': f'Could not load data for {site_name}'} 
                for month, _, _, _ in months_to_process]
    
    posts = data['posts']
    users = data['users']
    comments = data.get('comments', {})
    
    results = []
    
    # Process each month
    for month, month_idx, all_months, output_path_str in months_to_process:
        output_path = Path(output_path_str)
        
        # Skip if already exists
        if (site_name, month) in existing_files:
            results.append({'status': 'skipped', 'site_name': site_name, 'month': month})
            continue
        
        # Get adjacent months for temporal features
        prev_month = all_months[month_idx-1] if month_idx > 0 else None
        next_month = all_months[month_idx+1] if month_idx < len(all_months)-1 else None
        
        # Build graph
        try:
            graph = create_hetero_graph_with_features(posts, users, comments, month, prev_month, next_month)
            
            if graph is not None:
                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(graph, output_path)
                results.append({'status': 'built', 'site_name': site_name, 'month': month})
            else:
                # Graph is None - likely no data for this month
                results.append({'status': 'skipped', 'site_name': site_name, 'month': month})
        except Exception as e:
            import traceback
            error_msg = f"Error processing {site_name} {month}: {e}\n{traceback.format_exc()}"
            results.append({'status': 'error', 'site_name': site_name, 'month': month, 'error': error_msg})
    
    return results


def scan_existing_files(output_dir: Path) -> Set[Tuple[str, str]]:
    """
    Scan output directory to find existing (site, month) pairs.
    This is the authoritative source - scans the actual filesystem.
    
    Files are expected in format: output_dir/site_name/yyyy-mm.pt
    
    Returns:
        Set of (site_name, month) tuples that already have valid output files
    """
    print(f"Scanning for existing graph files in {output_dir}...")
    existing = set()
    
    if not output_dir.exists():
        print(f"  Output directory does not exist: {output_dir}")
        return existing
    
    # Scan each site directory
    site_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    print(f"  Found {len(site_dirs)} site directories")
    
    for site_dir in tqdm(site_dirs, desc="Scanning sites", unit="site"):
        site_name = site_dir.name
        
        # Find all .pt files in this site directory
        for graph_file in site_dir.glob("*.pt"):
            # Extract month from filename (e.g., "2020-01.pt" -> "2020-01")
            month = graph_file.stem
            
            # Validate YYYY-MM format
            if len(month) == 7 and month[4] == '-':
                try:
                    # Quick validation: year should be 4 digits, month 2 digits
                    year_part = month[:4]
                    month_part = month[5:7]
                    if year_part.isdigit() and month_part.isdigit():
                        # Check file has non-zero size (fast check)
                        if check_existing_file_fast(graph_file):
                            existing.add((site_name, month))
                except (ValueError, IndexError):
                    continue
    
    print(f"  Found {len(existing)} existing graph files")
    return existing


def process_single_site(
    site_folder_str: str,
    output_dir_str: str
) -> List[Tuple]:
    """
    Process a single site to collect all its month tasks.
    
    Returns:
        List of task tuples for this site, or empty list if site should be skipped
    """
    site_folder = Path(site_folder_str)
    output_dir = Path(output_dir_str)
    site_name = site_folder.name
    
    # Load data to get months
    data = load_site_data(site_folder)
    if data is None:
        return []
    
    posts = data['posts']
    months = get_sorted_months(posts)
    
    if not months:
        return []
    
    # Create output directory for this site
    site_output = output_dir / site_name
    site_output.mkdir(parents=True, exist_ok=True)
    
    # Build tasks for each month
    tasks = []
    for i, month in enumerate(months):
        output_path = site_output / f"{month}.pt"
        tasks.append((
            str(site_folder),
            site_name,
            month,
            i,
            months,
            str(output_path),
            True  # skip_existing flag (will be filtered before processing)
        ))
    
    return tasks


def get_cache_path() -> Path:
    """Get the path to the cache file for collected tasks."""
    return Path(__file__).parent / "collected_tasks_cache.pkl"


def load_cached_tasks(
    site_folders: List[Path],
    output_dir: Path
) -> Optional[List[Tuple]]:
    """
    Load cached tasks if they exist and match the current parameters.
    
    Returns:
        Cached tasks if valid, None otherwise
    """
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check if parameters match
        cached_site_folders = cached_data.get('site_folders')
        cached_output_dir = cached_data.get('output_dir')
        
        # Normalize paths for comparison
        current_site_folders = sorted([str(f.resolve()) for f in site_folders])
        cached_site_folders_normalized = sorted([str(Path(f).resolve()) for f in cached_site_folders])
        
        if (cached_site_folders_normalized == current_site_folders and 
            str(Path(cached_output_dir).resolve()) == str(output_dir.resolve())):
            print(f"Loading cached tasks from {cache_path}")
            return cached_data.get('tasks')
        else:
            print("Cache parameters don't match, will regenerate tasks")
            return None
    except Exception as e:
        print(f"Error loading cache: {e}, will regenerate tasks")
        return None


def save_cached_tasks(
    tasks: List[Tuple],
    site_folders: List[Path],
    output_dir: Path
):
    """Save tasks to cache file."""
    cache_path = get_cache_path()
    try:
        cached_data = {
            'tasks': tasks,
            'site_folders': [str(f) for f in site_folders],
            'output_dir': str(output_dir)
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Saved {len(tasks)} tasks to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def collect_all_tasks(
    site_folders: List[Path],
    output_dir: Path,
    num_workers: int
) -> List[Tuple]:
    """
    Collect all (site, month) tasks across all sites in parallel.
    
    Returns:
        List of tuples: (site_folder_str, site_name, month, month_idx, all_months, output_path_str, skip_existing)
    """
    # Try to load from cache first
    cached_tasks = load_cached_tasks(site_folders, output_dir)
    if cached_tasks is not None:
        return cached_tasks
    
    print("Collecting tasks from all sites...")
    
    # Prepare arguments for parallel processing
    site_args = [(str(site_folder), str(output_dir)) for site_folder in site_folders]
    
    # Process sites in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.starmap(process_single_site, site_args),
            total=len(site_args),
            desc="Loading site data",
            unit="site"
        ))
    
    # Flatten results into single list
    all_tasks = []
    for site_tasks in results:
        all_tasks.extend(site_tasks)
    
    # Save to cache
    save_cached_tasks(all_tasks, site_folders, output_dir)
    
    return all_tasks


def filter_existing_tasks(
    all_tasks: List[Tuple],
    existing: Set[Tuple[str, str]]
) -> List[Tuple]:
    """
    Filter out tasks that already have output files.
    
    Returns:
        List of tasks that need to be processed
    """
    filtered = []
    for task in all_tasks:
        site_name = task[1]
        month = task[2]
        if (site_name, month) not in existing:
            # Update skip_existing to False since we've already filtered
            task_list = list(task)
            task_list[6] = False  # skip_existing is at index 6
            filtered.append(tuple(task_list))
    return filtered


def group_tasks_by_site(all_tasks: List[Tuple]) -> Dict[str, List[Tuple]]:
    """
    Group tasks by site name so we can process all months for a site together.
    
    Returns:
        Dictionary mapping site_name to list of task tuples
    """
    grouped = defaultdict(list)
    for task in all_tasks:
        site_name = task[1]  # site_name is at index 1
        grouped[site_name].append(task)
    return dict(grouped)


def process_all_tasks_parallel(
    all_tasks: List[Tuple],
    existing_files: Set[Tuple[str, str]],
    num_workers: int
) -> Dict:
    """
    Process all (site, month) tasks in parallel, grouping by site to minimize file I/O.
    Each worker processes all months for a site after loading the data once.
    
    Returns:
        Dictionary with aggregate statistics
    """
    total_stats = {'built': 0, 'skipped': 0, 'errors': 0}
    error_messages = []
    skipped_combos = []  # Track skipped community/month combinations
    
    # Group tasks by site
    grouped_tasks = group_tasks_by_site(all_tasks)
    print(f"\nProcessing {len(all_tasks)} tasks across {len(grouped_tasks)} sites using {num_workers} workers...")
    print("Grouping by site to minimize file I/O (load once per site, process all months)")
    
    # Prepare arguments for grouped processing
    # Each task is: (site_folder_str, site_name, months_to_process, existing_files)
    site_args = []
    for site_name, site_tasks in grouped_tasks.items():
        site_folder_str = site_tasks[0][0]  # All tasks for a site have the same folder
        # Extract month info: (month, month_idx, all_months, output_path_str)
        months_to_process = [(task[2], task[3], task[4], task[5]) for task in site_tasks]
        site_args.append((site_folder_str, site_name, months_to_process, existing_files))
    
    # Process sites in parallel (each worker processes all months for its assigned site)
    with Pool(processes=num_workers) as pool:
        result_lists = list(tqdm(
            pool.starmap(process_site_months, site_args),
            total=len(site_args),
            desc="Building graphs",
            unit="site"
        ))
    
    # Flatten and aggregate results
    for result_list in result_lists:
        for result in result_list:
            if result['status'] == 'built':
                total_stats['built'] += 1
            elif result['status'] == 'skipped':
                total_stats['skipped'] += 1
                skipped_combos.append((result['site_name'], result['month']))
            elif result['status'] == 'error':
                total_stats['errors'] += 1
                if 'error' in result:
                    error_messages.append(result['error'])
    
    # Print errors if any
    if error_messages:
        print(f"\n{'='*60}")
        print(f"Errors encountered ({len(error_messages)}):")
        print(f"{'='*60}")
        for error_msg in error_messages[:10]:  # Print first 10 errors
            print(error_msg)
        if len(error_messages) > 10:
            print(f"... and {len(error_messages) - 10} more errors")
    
    # Store skipped combos in stats for printing
    total_stats['skipped_combos'] = skipped_combos
    
    return total_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Build PyG graphs for Stack Exchange sites')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing parsed site data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save built graphs')
    parser.add_argument('--sites', type=str, nargs='+', default=None, help='Specific sites to process')
    parser.add_argument('--no-skip-existing', action='store_true', help='Rebuild graphs even if they already exist')
    parser.add_argument('--num-workers', type=int, default=None, help=f'Number of parallel workers (default: all CPU cores, {cpu_count()})')
    return parser.parse_args()


def get_site_folders(data_dir: Path, sites: Optional[List[str]]) -> List[Path]:
    """Get list of site folders to process."""
    if sites:
        return [data_dir / site for site in sites]
    return [f for f in data_dir.iterdir() if f.is_dir()]


def print_summary(total_stats: Dict, output_dir: Path, initial_count: int, filtered_count: int):
    """Print final summary."""
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total tasks found: {initial_count}")
    print(f"Already completed: {initial_count - filtered_count}")
    print(f"Tasks processed: {filtered_count}")
    print(f"  - Built: {total_stats['built']}")
    print(f"  - Skipped: {total_stats['skipped']}")
    print(f"  - Errors: {total_stats['errors']}")
    print(f"Graphs saved to: {output_dir}")
    
    # Print skipped community/month combinations
    skipped_combos = total_stats.get('skipped_combos', [])
    if skipped_combos:
        print(f"\n{'='*60}")
        print(f"Skipped Community/Month Combinations ({len(skipped_combos)}):")
        print(f"{'='*60}")
        # Sort for easier reading
        skipped_combos_sorted = sorted(skipped_combos)
        for site_name, month in skipped_combos_sorted:
            print(f"  {site_name} / {month}")


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
    
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"Found {len(site_folders)} sites to process")
    print(f"Using {num_workers} parallel workers")
    
    # Collect all tasks across all sites
    all_tasks = collect_all_tasks(site_folders, output_dir, num_workers)
    
    if not all_tasks:
        sys.exit("Error: No tasks to process")
    
    initial_task_count = len(all_tasks)
    print(f"\nCollected {initial_task_count} total tasks")
    
    # Check existing files by scanning the filesystem (unless --no-skip-existing is set)
    existing = set()
    if not args.no_skip_existing:
        existing = scan_existing_files(output_dir)
        print(f"Found {len(existing)} already-completed tasks")
        
        # Filter out existing tasks
        all_tasks = filter_existing_tasks(all_tasks, existing)
        print(f"Filtered to {len(all_tasks)} tasks to process")
    else:
        print("Skipping existing file check (--no-skip-existing set)")
        # Update all tasks to not skip existing
        all_tasks = [tuple(list(task)[:6] + [False]) for task in all_tasks]
    
    if not all_tasks:
        print("All tasks already completed!")
        return
    
    # Process all tasks in parallel, grouped by site to minimize file I/O
    total_stats = process_all_tasks_parallel(all_tasks, existing, num_workers)
    print_summary(total_stats, output_dir, initial_task_count, len(all_tasks))


if __name__ == "__main__":
    main()