"""
Parses Stack Exchange metadata XML files: Tags.xml, Users.xml, and Comments.xml.
Returns structured data for graph enrichment.

Usage:
    from data_processing.parse_metadata import parse_tags_xml, parse_users_xml, parse_comments_xml
    
    tags = parse_tags_xml('path/to/Tags.xml')
    users = parse_users_xml('path/to/Users.xml')
    comments = parse_comments_xml('path/to/Comments.xml')
"""

import xml.etree.ElementTree as ET
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
from pathlib import Path


def parse_tags_xml(xml_path: str) -> Dict[str, Dict]:
    """
    Parse Tags.xml file.
    
    Args:
        xml_path: Path to Tags.xml file
        
    Returns:
        Dictionary mapping tag names to metadata:
        {
            'python': {
                'tag_id': '1',
                'count': 5000
            },
            ...
        }
    """
    print(f"Parsing {xml_path}...")
    
    tags = {}
    tag_count = 0
    
    # Get total file size for progress bar
    file_size = Path(xml_path).stat().st_size

    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Parsing Tags") as pbar:
        context = ET.iterparse(xml_path, events=('end',))
    
        for event, elem in context:
            # Update progress bar
            if elem.tag == 'row':
                pbar.update(len(ET.tostring(elem)))

            if elem.tag != 'row':
                continue
            
            tag_name = elem.get('TagName')
            if not tag_name:
                elem.clear()
                continue
            
            tags[tag_name] = {
                'tag_id': elem.get('Id'),
                'count': int(elem.get('Count', 0))
            }
            
            tag_count += 1
            elem.clear()
    
    print(f"  Tags parsed: {tag_count:,}")
    return tags


def parse_users_xml(xml_path: str) -> Dict[str, Dict]:
    """
    Parse Users.xml file.
    
    Args:
        xml_path: Path to Users.xml file
        
    Returns:
        Dictionary mapping user IDs to metadata:
        {
            '456': {
                'reputation': 1234,
                'creation_date': '2019-05-10T08:00:00.000',
                'last_access_date': '2020-01-20T14:00:00.000',
                'display_name': 'JohnDoe',
                'location': 'San Francisco, CA',
                'views': 100,
                'upvotes': 50,
                'downvotes': 5,
                'account_id': '34933'
            },
            ...
        }
    """
    print(f"Parsing {xml_path}...")
    
    users = {}
    user_count = 0

    # Get total file size for progress bar
    file_size = Path(xml_path).stat().st_size

    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Parsing Users") as pbar:
        context = ET.iterparse(xml_path, events=('end',))
    
        for event, elem in context:
            # Update progress bar
            if elem.tag == 'row':
                pbar.update(len(ET.tostring(elem)))
    
            if elem.tag != 'row':
                continue
            
            user_id = elem.get('Id')
            if not user_id:
                elem.clear()
                continue
            
            users[user_id] = {
                'reputation': int(elem.get('Reputation', 0)),
                'creation_date': elem.get('CreationDate'),
                'last_access_date': elem.get('LastAccessDate'),
                'display_name': elem.get('DisplayName'),
                'location': elem.get('Location'),
                'views': int(elem.get('Views', 0)),
                'upvotes': int(elem.get('UpVotes', 0)),
                'downvotes': int(elem.get('DownVotes', 0)),
                'account_id': elem.get('AccountId')
            }
            
            user_count += 1
            elem.clear()
    
    print(f"  Users parsed: {user_count:,}")
    return users


def parse_comments_xml(xml_path: str) -> List[Dict]:
    """
    Parse Comments.xml file.
    
    Args:
        xml_path: Path to Comments.xml file
        
    Returns:
        List of comment dictionaries:
        [
            {
                'comment_id': '35',
                'post_id': '39',
                'user_id': '232',
                'score': 3,
                'creation_date': '2018-01-18T03:07:30.307'
            },
            ...
        ]
    """
    print(f"Parsing {xml_path}...")
    
    comments = []
    comment_count = 0

    # Get total file size for progress bar
    file_size = Path(xml_path).stat().st_size

    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Parsing Comments") as pbar:
        context = ET.iterparse(xml_path, events=('end',))
    
        for event, elem in context:
            # Update progress bar
            if elem.tag == 'row':
                pbar.update(len(ET.tostring(elem)))
    
            if elem.tag != 'row':
                continue
            
            comment = {
                'comment_id': elem.get('Id'),
                'post_id': elem.get('PostId'),
                'user_id': elem.get('UserId'),
                'score': int(elem.get('Score', 0)),
                'creation_date': elem.get('CreationDate')
            }
            
            comments.append(comment)
            comment_count += 1
            elem.clear()
    
    print(f"  Comments parsed: {comment_count:,}")
    return comments


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Usage:")
    print("  from data_processing.parse_metadata import parse_tags_xml, parse_users_xml, parse_comments_xml")
    print("  tags = parse_tags_xml('path/to/Tags.xml')")
    print("  users = parse_users_xml('path/to/Users.xml')")
    print("  comments = parse_comments_xml('path/to/Comments.xml')")