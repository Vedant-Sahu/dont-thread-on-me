"""
Feature extraction functions for tags and users in Stack Exchange data.

Provides functions to compute various features for tags and users that are used
in graph construction.

Usage:
    from data_processing.feature_extraction import build_tag_features, build_user_features
    
    tag_features = build_tag_features(posts, month, previous_month)
    user_features = build_user_features(posts, users, month, next_month)
"""

import numpy as np
from calendar import monthrange
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional
from scipy.stats import entropy


# ===== TAG FEATURES =====

def compute_tag_popularity(posts: Dict, month: str, tag: str) -> int:
    """Number of questions with this tag in the month."""
    questions = posts[month]['questions']
    count = sum(1 for q in questions if tag in q['tags'])
    return count


def compute_tag_growth_rate(posts: Dict, current_month: str, previous_month: str, tag: str) -> float:
    """Growth rate = (current - previous) / previous."""
    if previous_month not in posts:
        return 0.0
    
    curr_pop = compute_tag_popularity(posts, current_month, tag)
    prev_pop = compute_tag_popularity(posts, previous_month, tag)
    
    if prev_pop == 0:
        return 0.0 if curr_pop == 0 else 1.0
    
    return (curr_pop - prev_pop) / prev_pop


def compute_tag_answer_quality(posts: Dict, month: str, tag: str) -> float:
    """Average score of answers to questions with this tag."""
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    # Get question IDs with this tag
    question_ids = {q['post_id'] for q in questions if tag in q['tags']}
    
    # Get answers to those questions
    relevant_answers = [a for a in answers if a['parent_id'] in question_ids]
    
    if not relevant_answers:
        return 0.0
    
    avg_score = np.mean([a['score'] for a in relevant_answers])
    return float(avg_score)


def compute_tag_difficulty(posts: Dict, month: str, tag: str) -> float:
    """Difficulty = fraction of questions with no accepted answer."""
    questions = posts[month]['questions']
    tag_questions = [q for q in questions if tag in q['tags']]
    
    if not tag_questions:
        return 0.0
    
    unanswered = sum(1 for q in tag_questions if not q['accepted_answer_id'])
    return unanswered / len(tag_questions)


def compute_tag_diversity(posts: Dict, month: str, tag: str) -> float:
    """Entropy of user distribution: how many different users contribute to this tag."""
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    # Count contributions per user for this tag
    user_counts = defaultdict(int)
    
    for q in questions:
        if tag in q['tags'] and q['user_id']:
            user_counts[q['user_id']] += 1
    
    for a in answers:
        if tag in a['parent_tags'] and a['user_id']:
            user_counts[a['user_id']] += 1
    
    if not user_counts:
        return 0.0
    
    counts = list(user_counts.values())
    return float(entropy(counts))  # Higher = more diverse


def build_tag_features(posts: Dict, month: str, previous_month: Optional[str] = None) -> Dict[str, Dict]:
    """Build feature dictionary for all tags in a month."""
    questions = posts[month]['questions']
    
    # Get all tags
    tags = set()
    for q in questions:
        tags.update(q['tags'])
    
    tag_features = {}
    
    for tag in tags:
        features = {
            'popularity': compute_tag_popularity(posts, month, tag),
            'answer_quality': compute_tag_answer_quality(posts, month, tag),
            'difficulty': compute_tag_difficulty(posts, month, tag),
            'diversity': compute_tag_diversity(posts, month, tag),
        }
        
        # Add growth rate if previous month available
        if previous_month:
            features['growth_rate'] = compute_tag_growth_rate(posts, month, previous_month, tag)
        else:
            features['growth_rate'] = 0.0
        
        tag_features[tag] = features
    
    return tag_features


# ===== USER FEATURES ===== 

def compute_user_reputation(users: Dict, user_id: str) -> int:
    """Get user reputation from users dict."""
    if not users or user_id not in users:
        return 0
    return users[user_id].get('reputation', 0)


def compute_user_tenure(users: Dict, user_id: str, current_month: str) -> int:
    """Months since user joined (in months)."""
    if not users or user_id not in users:
        return 0
    
    creation_date = users[user_id].get('creation_date')
    if not creation_date:
        return 0
    
    # Parse dates
    user_join = datetime.strptime(creation_date[:7], '%Y-%m')  # 'YYYY-MM'
    current = datetime.strptime(current_month, '%Y-%m')
    
    # Calculate month difference
    months_diff = (current.year - user_join.year) * 12 + (current.month - user_join.month)
    return max(0, months_diff)


def compute_user_activity(posts: Dict, month: str, user_id: str) -> int:
    """Total contributions (questions + answers) in the month."""
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    q_count = sum(1 for q in questions if q['user_id'] == user_id)
    a_count = sum(1 for a in answers if a['user_id'] == user_id)
    
    return q_count + a_count


def compute_user_expertise_entropy(posts: Dict, month: str, user_id: str) -> float:
    """Entropy of tag distribution: how specialized vs generalist."""
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    tag_counts = defaultdict(int)
    
    for q in questions:
        if q['user_id'] == user_id:
            for tag in q['tags']:
                tag_counts[tag] += 1
    
    for a in answers:
        if a['user_id'] == user_id:
            for tag in a['parent_tags']:
                tag_counts[tag] += 1
    
    if not tag_counts:
        return 0.0
    
    counts = list(tag_counts.values())
    return float(entropy(counts))  # Higher = more generalist


def compute_user_retention(posts: Dict, month: str, next_month: str, user_id: str) -> int:
    """Binary: is user active in next month?"""
    if next_month not in posts:
        return 0
    
    next_questions = posts[next_month]['questions']
    next_answers = posts[next_month]['answers']
    
    active_next = any(q['user_id'] == user_id for q in next_questions) or \
                  any(a['user_id'] == user_id for a in next_answers)
    
    return 1 if active_next else 0


def build_user_features(posts: Dict, users: Dict, month: str, next_month: Optional[str] = None) -> Dict[str, Dict]:
    """Build feature dictionary for all users in a month."""
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    # Get all users
    user_ids = set()
    for q in questions:
        if q['user_id']:
            user_ids.add(q['user_id'])
    for a in answers:
        if a['user_id']:
            user_ids.add(a['user_id'])
    
    user_features = {}
    
    for user_id in user_ids:
        features = {
            'reputation': compute_user_reputation(users, user_id),
            'tenure': compute_user_tenure(users, user_id, month),
            'activity': compute_user_activity(posts, month, user_id),
            'expertise_entropy': compute_user_expertise_entropy(posts, month, user_id),
        }
        
        # Add retention if next month available
        if next_month:
            features['retention'] = compute_user_retention(posts, month, next_month, user_id)
        else:
            features['retention'] = 0
        
        user_features[user_id] = features
    
    return user_features


# ===== COMMUNITY METRICS (Graph-level targets) =====

def compute_community_metrics(
    posts: Dict,
    users: Dict,
    month: str,
    prev_month: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """
    Compute community-level health metrics for a given month.
    
    These metrics serve as prediction targets when this month appears
    6 months after a training sequence.
    
    Args:
        posts: Dictionary of monthly post data
        users: Dictionary of user data
        month: Current month (YYYY-MM format)
        prev_month: Previous month for computing growth rate
    
    Returns:
        Dictionary with 4 metrics: qpd, answer_rate, retention, growth
        Returns None if insufficient data
    """
    if month not in posts:
        return None
    
    questions = posts[month]['questions']
    answers = posts[month]['answers']
    
    if not questions:
        return None
    
    # 1. Questions per day (QPD)
    # Parse year and month from format 'YYYY-MM'
    year, month_num = map(int, month.split('-'))
    # Get exact number of days in the month
    days_in_month = monthrange(year, month_num)[1]
    qpd = len(questions) / days_in_month
    
    # 2. Answer rate (fraction of questions with accepted answers)
    questions_with_accepted = sum(1 for q in questions if q.get('accepted_answer_id') is not None)
    answer_rate = questions_with_accepted / len(questions) if questions else 0.0
    
    # 3. User retention (average retention rate of active users)
    active_users_this_month = set()
    for q in questions:
        if q.get('user_id'):
            active_users_this_month.add(q['user_id'])
    for a in answers:
        if a.get('user_id'):
            active_users_this_month.add(a['user_id'])
    
    retention = 0.0
    if active_users_this_month:
        user_features = build_user_features(posts, users, month, None)
        retention_values = [user_features[uid]['retention'] for uid in active_users_this_month 
                           if uid in user_features]
        retention = sum(retention_values) / len(retention_values) if retention_values else 0.0
    
    # 4. New user growth rate (percentage change in new user count)
    all_months = sorted([m for m in posts.keys() if m and m != 'metadata' and m <= month])
    
    current_month_users = set()
    for q in questions:
        if q.get('user_id'):
            current_month_users.add(q['user_id'])
    for a in answers:
        if a.get('user_id'):
            current_month_users.add(a['user_id'])
    
    # Find new users (no activity in previous months)
    if len(all_months) > 1:
        prev_months = all_months[:-1]
        previously_active = set()
        
        for prev_m in prev_months:
            if prev_m in posts:
                for q in posts[prev_m].get('questions', []):
                    if q.get('user_id'):
                        previously_active.add(q['user_id'])
                for a in posts[prev_m].get('answers', []):
                    if a.get('user_id'):
                        previously_active.add(a['user_id'])
        
        new_users_this_month = current_month_users - previously_active
    else:
        new_users_this_month = current_month_users
    
    # Compute growth rate relative to previous month
    growth = 0.0
    if prev_month and prev_month in posts:
        prev_questions = posts[prev_month].get('questions', [])
        prev_answers = posts[prev_month].get('answers', [])
        
        prev_month_users = set()
        for q in prev_questions:
            if q.get('user_id'):
                prev_month_users.add(q['user_id'])
        for a in prev_answers:
            if a.get('user_id'):
                prev_month_users.add(a['user_id'])
        
        months_before_prev = [m for m in all_months if m < prev_month]
        previously_active_before_prev = set()
        
        for m in months_before_prev:
            if m in posts:
                for q in posts[m].get('questions', []):
                    if q.get('user_id'):
                        previously_active_before_prev.add(q['user_id'])
                for a in posts[m].get('answers', []):
                    if a.get('user_id'):
                        previously_active_before_prev.add(a['user_id'])
        
        new_users_prev_month = prev_month_users - previously_active_before_prev
        
        if len(new_users_prev_month) > 0:
            growth = (len(new_users_this_month) - len(new_users_prev_month)) / len(new_users_prev_month)
        elif len(new_users_this_month) > 0:
            growth = 1.0
    
    return {
        'qpd': float(qpd),
        'answer_rate': float(answer_rate),
        'retention': float(retention),
        'growth': float(growth)
    }