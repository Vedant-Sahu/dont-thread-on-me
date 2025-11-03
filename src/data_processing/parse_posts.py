"""
Parses Stack Exchange Posts.xml file and groups posts by month.
Extracts questions and answers with relevant metadata for graph construction.

Usage:
    from data_processing.parse_posts import parse_posts_xml
    monthly_data = parse_posts_xml('path/to/Posts.xml')
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


def parse_tags(tags_string: str) -> List[str]:
    """
    Parse pipe-delimited tags string into list of tag names.
    
    Args:
        tags_string: Pipe-delimited tags like "|python|pandas|data|"
        
    Returns:
        List of tag names like ['python', 'pandas', 'data']
    """
    if not tags_string:
        return []
    
    # Remove leading/trailing pipes and split
    tags = tags_string.strip('|').split('|')
    
    # Filter out empty strings
    return [tag for tag in tags if tag]


def extract_month(date_string: str) -> str:
    """
    Extract YYYY-MM from ISO datetime string.
    
    Args:
        date_string: ISO format like "2020-01-15T10:30:00.000"
        
    Returns:
        Month string like "2020-01"
    """
    if not date_string:
        return None
    try:
        # Simply take the first 7 characters (YYYY-MM)
        return date_string[:7]
    except:
        return None
    

def parse_posts_xml(xml_path: str) -> Dict[str, Dict]:
    """
    Parse Posts.xml file and group by month using two-pass approach.
    
    Pass 1: Parse all questions and build lookup table
    Pass 2: Parse answers and enrich with parent question info
    
    Args:
        xml_path: Path to Posts.xml file
        
    Returns:
        Dictionary structured as:
        {
            '2020-01': {
                'questions': [
                    {
                        'post_id': str,
                        'user_id': str,
                        'tags': List[str],
                        'score': int,
                        'view_count': int,
                        'answer_count': int,
                        'accepted_answer_id': Optional[str],
                        'creation_date': str
                    },
                    ...
                ],
                'answers': [
                    {
                        'post_id': str,
                        'parent_id': str,
                        'parent_tags': List[str],  
                        'user_id': str,
                        'score': int,
                        'creation_date': str
                    },
                    ...
                ]
            },
            '2020-02': {...}
        }
    """
    print(f"Parsing {xml_path}...")

    # Get total file size for progress bar
    file_size = Path(xml_path).stat().st_size
    
    # Pass 1: Build question lookup
    question_lookup = {}  # {question_id: (month, tags)}
    monthly_data = defaultdict(lambda: {'questions': [], 'answers': []})
    
    question_count = 0
    skipped_count = 0

    # Create progress bar based on bytes processed
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Pass 1") as pbar:
        context = ET.iterparse(xml_path, events=('end',))
    
        for event, elem in context:
            # Update progress bar
            if elem.tag == 'row':
                pbar.update(len(ET.tostring(elem)))
            
            if elem.tag != 'row':
                continue
            
            post_type = elem.get('PostTypeId')
            
            # Only process questions in Pass 1
            if post_type == '1':
                creation_date = elem.get('CreationDate')
                month = extract_month(creation_date)
                
                if not month:
                    skipped_count += 1
                    elem.clear()
                    continue
                
                post_id = elem.get('Id')
                tags = parse_tags(elem.get('Tags', ''))
                
                # Store in lookup table
                question_lookup[post_id] = (month, tags)
                
                # Create question object
                question = {
                    'post_id': post_id,
                    'user_id': elem.get('OwnerUserId'),
                    'tags': tags,
                    'score': int(elem.get('Score', 0)),
                    'view_count': int(elem.get('ViewCount', 0)),
                    'answer_count': int(elem.get('AnswerCount', 0)),
                    'accepted_answer_id': elem.get('AcceptedAnswerId'),
                    'creation_date': creation_date
                }
                
                monthly_data[month]['questions'].append(question)
                question_count += 1
            
            elif post_type not in ('1', '2'):
                skipped_count += 1
            
            elem.clear()
        
    print(f"  Questions parsed: {question_count:,}")
    print(f"  Questions in lookup: {len(question_lookup):,}")
    
    # Pass 2: Parse answers and enrich with parent info    
    answer_count = 0
    orphaned_answers = 0

    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Pass 2") as pbar:
        context = ET.iterparse(xml_path, events=('end',))
    
        for event, elem in context  :
            # Update progress bar
            if elem.tag == 'row':
                pbar.update(len(ET.tostring(elem)))
            if elem.tag != 'row':
                continue
            
            post_type = elem.get('PostTypeId')
            
            # Only process answers in Pass 2
            if post_type == '2':
                creation_date = elem.get('CreationDate')
                month = extract_month(creation_date)
                
                if not month:
                    elem.clear()
                    continue
                
                parent_id = elem.get('ParentId')
                
                # Look up parent question's tags
                parent_info = question_lookup.get(parent_id)
                
                if not parent_info:
                    # Parent question doesn't exist (deleted or missing)
                    orphaned_answers += 1
                    elem.clear()
                    continue
                
                parent_month, parent_tags = parent_info
                
                answer = {
                    'post_id': elem.get('Id'),
                    'parent_id': parent_id,
                    'parent_month': parent_month,
                    'parent_tags': parent_tags, 
                    'user_id': elem.get('OwnerUserId'),
                    'score': int(elem.get('Score', 0)),
                    'creation_date': creation_date
                }
                
                monthly_data[month]['answers'].append(answer)
                answer_count += 1
            
            elem.clear()
    
    print(f"  Answers parsed: {answer_count:,}")
    print(f"  Orphaned answers (no parent): {orphaned_answers:,}")
    print(f"  Skipped (other post types): {skipped_count:,}")
    print(f"  Months covered: {len(monthly_data)}")
    
    return dict(monthly_data)


def get_monthly_summary(monthly_data: Dict[str, Dict]) -> None:
    """
    Print summary statistics for parsed monthly data.
    
    Args:
        monthly_data: Output from parse_posts_xml()
    """
    print("\n" + "="*60)
    print("Monthly Summary")
    print("="*60)
    
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        q_count = len(data['questions'])
        a_count = len(data['answers'])
        
        # Get unique tags for this month
        all_tags = set()
        for q in data['questions']:
            all_tags.update(q['tags'])
        
        print(f"{month}: {q_count:>4} questions, {a_count:>4} answers, {len(all_tags):>3} tags")


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Usage:")
    print("  from data_preprocessing.parse_posts import parse_posts_xml")
    print("  monthly_data = parse_posts_xml('path/to/Posts.xml')")