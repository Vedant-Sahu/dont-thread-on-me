"""
Enumerates all Stack Exchange sites available in the Archive.org Stack Exchange data dump.
Retrieves metadata about each site and saves it to a JSON file for later analysis and selection.

Usage:
    python src/data_collection/enumerate_sites.py

Output:
    - data/processed/site_metadata.json: Comprehensive metadata about all available sites
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import internetarchive as ia
from tqdm import tqdm


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
STACKEXCHANGE_IDENTIFIER = "stackexchange"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "site_metadata.json"


def get_stackexchange_item() -> ia.Item:
    """
    Get the Stack Exchange Archive.org item.
    
    Returns:
        ia.Item: The Stack Exchange item from Archive.org
    """
    print(f"Fetching Stack Exchange item from Archive.org...")
    item = ia.get_item(STACKEXCHANGE_IDENTIFIER)
    
    if not item.exists:
        raise ValueError(f"Item '{STACKEXCHANGE_IDENTIFIER}' does not exist on Archive.org")
    
    return item


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse a Stack Exchange filename to extract site information.
    
    Stack Exchange files follow two patterns:
        1. Complete site archive: sitename.7z (e.g., physics.meta.stackexchange.com.7z)
        2. Individual table files: sitename-TableName.7z (e.g., stackoverflow.com-Posts.7z)
    
    Filters out:
    - Meta sites (e.g., stats.meta.stackexchange.com.7z)
    - Non-English language versions (e.g., ru.stackoverflow.com)
    
    Args:
        filename: The name of the file
        
    Returns:
        Dict with 'site_name', 'table_type', and 'extension', or None if not a data file
    """
    # Only process .7z files (the data files)
    if not filename.endswith('.7z'):
        return None
    
    # Remove the extension
    name_without_ext = filename[:-3]

    # Filter out meta sites
    if '.meta.' in name_without_ext or name_without_ext.startswith('meta.'):
        return None
    
    # Filter out non-English language versions (ru, pt, es, ja, etc.)
    language_prefixes = ['ru.', 'pt.', 'es.', 'ja.']
    if any(name_without_ext.startswith(prefix) for prefix in language_prefixes):
        return None

    # Filter out non-table files (like metadata files)
    valid_tables = ['Badges', 'Comments', 'PostHistory', 'PostLinks', 'Posts', 
                    'Users', 'Tags', 'Votes', ]
    
    # Check if it's an individual table file (pattern: sitename-TableName)
    if '-' in name_without_ext:
        parts = name_without_ext.rsplit('-', 1)
        if len(parts) == 2:
            site_name, table_type = parts
            if table_type in valid_tables:
                return {
                    'site_name': site_name,
                    'table_type': table_type,
                    'extension': '.7z',
                    'file_type': 'individual_table'
                }
    
    # Otherwise, it's a complete site archive (pattern: sitename.7z)
    # Contains all tables in one file
    return {
        'site_name': name_without_ext,
        'table_type': 'complete_archive',
        'extension': '.7z',
        'file_type': 'complete_archive'
    }


def enumerate_sites(item: ia.Item) -> Dict[str, Dict]:
    """
    Enumerate all Stack Exchange sites from the Archive.org item.
    
    Args:
        item: The Archive.org item containing Stack Exchange data
        
    Returns:
        Dict mapping site names to their metadata
    """
    print("Enumerating Stack Exchange sites...")
    
    sites = {}
    
    # Get all files from the item
    files = list(item.get_files())
    print(f"Found {len(files)} total files in the Stack Exchange archive")
    
    # Process each file
    for file in tqdm(files, desc="Processing files"):
        parsed = parse_filename(file.name)
        
        if parsed is None:
            continue
        
        site_name = parsed['site_name']
        table_type = parsed['table_type']
        
        # Initialize site entry if not exists
        if site_name not in sites:
            sites[site_name] = {
                'site_name': site_name,
                'tables': {},
                'total_size_bytes': 0,
                'file_count': 0,
                'last_modified': None
            }
        
        # Add table information
        sites[site_name]['tables'][table_type] = {
            'filename': file.name,
            'size_bytes': int(file.size) if file.size else 0,
            'format': file.format if hasattr(file, 'format') else 'Unknown',
            'md5': file.md5 if hasattr(file, 'md5') else None,
            'mtime': file.mtime if hasattr(file, 'mtime') else None
        }
        
        # Update site totals
        sites[site_name]['total_size_bytes'] += int(file.size) if file.size else 0
        sites[site_name]['file_count'] += 1
        
        # Track most recent modification time
        if hasattr(file, 'mtime') and file.mtime:
            if sites[site_name]['last_modified'] is None or file.mtime > sites[site_name]['last_modified']:
                sites[site_name]['last_modified'] = file.mtime
    
    print(f"\nFound {len(sites)} unique Stack Exchange sites")
    
    return sites


def add_site_statistics(sites: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Add computed statistics to site metadata.
    
    Args:
        sites: Dictionary of site metadata
        
    Returns:
        Updated sites dictionary with statistics
    """
    print("\nComputing site statistics...")
    
    for site_name, site_data in sites.items():
        # Convert bytes to MB for readability
        site_data['total_size_mb'] = round(site_data['total_size_bytes'] / (1024 * 1024), 2)
                
        # Categorize by size (rough estimates for community size)
        size_mb = site_data['total_size_mb']
        if size_mb < 50:
            site_data['size_category'] = 'small'
        elif size_mb < 500:
            site_data['size_category'] = 'medium'
        else:
            site_data['size_category'] = 'large'
    
    return sites


def save_metadata(sites: Dict[str, Dict], output_file: Path):
    """
    Save site metadata to a JSON file.
    
    Args:
        sites: Dictionary of site metadata
        output_file: Path to output JSON file
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata about the enumeration itself
    output_data = {
        'enumeration_date': datetime.now().isoformat(),
        'total_sites': len(sites),
        'sites': sites
    }
    
    # Add summary statistics
    total_size_gb = sum(site['total_size_bytes'] for site in sites.values()) / (1024**3)
    size_categories = {'small': 0, 'medium': 0, 'large': 0}
    for site in sites.values():
        size_categories[site['size_category']] += 1
    
    output_data['summary'] = {
        'total_size_gb': round(total_size_gb, 2),
        'size_distribution': size_categories
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nMetadata saved to: {output_file}")
    print(f"Total sites: {output_data['total_sites']}")
    print(f"Total size: {output_data['summary']['total_size_gb']:.2f} GB")
    print(f"Size distribution: {output_data['summary']['size_distribution']}")


def print_sample_sites(sites: Dict[str, Dict], n: int = 5):
    """
    Print information about a sample of sites.
    
    Args:
        sites: Dictionary of site metadata
        n: Number of sites to display
    """
    print(f"\n{'='*80}")
    print(f"Sample of {n} sites:")
    print(f"{'='*80}")
    
    # Sort by size and take top n
    sorted_sites = sorted(sites.items(), key=lambda x: x[1]['total_size_mb'], reverse=True)
    
    for i, (site_name, site_data) in enumerate(sorted_sites[:n], 1):
        print(f"\n{i}. {site_name}")
        print(f"   Size: {site_data['total_size_mb']:.2f} MB ({site_data['size_category']})")


def main():
    """Main execution function."""
    print("="*80)
    print("Stack Exchange Site Enumeration")
    print("="*80)
    
    try:
        # Get the Stack Exchange item from Archive.org
        item = get_stackexchange_item()
        
        # Enumerate all sites
        sites = enumerate_sites(item)
        
        # Add statistics
        sites = add_site_statistics(sites)
        
        # Save to JSON
        save_metadata(sites, OUTPUT_FILE)
        
        # Print sample
        print_sample_sites(sites, n=5)
        
        print(f"\n{'='*80}")
        print("Enumeration complete!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()