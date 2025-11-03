"""
Downloads Stack Exchange data from Archive.org, extracts relevant XML files,
parses them, and saves as compressed pickles.

Usage:
    python src/data_collection/download_pipeline.py
"""

import json
import gzip
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

import internetarchive as ia
import py7zr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.parse_posts import parse_posts_xml
from src.data_processing.parse_metadata import parse_tags_xml, parse_users_xml, parse_comments_xml

# Constants
STACKEXCHANGE_IDENTIFIER = "stackexchange"
SITE_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "site_metadata.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PARSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "parsed"

# XML files we need
REQUIRED_XMLS = ['Posts.xml', 'Tags.xml', 'Users.xml', 'Comments.xml']


def load_site_metadata() -> Dict:
    """Load site metadata from JSON file."""
    print(f"Loading site metadata from {SITE_METADATA_PATH}")
    
    if not SITE_METADATA_PATH.exists():
        print(f"Error: {SITE_METADATA_PATH} not found!")
        print("Please run enumerate_sites.py first.")
        sys.exit(1)
    
    with open(SITE_METADATA_PATH, 'r') as f:
        data = json.load(f)
    
    return data['sites']


def get_processed_sites() -> Set[str]:
    """
    Check which sites have already been processed.
    
    Returns:
        Set of site names that have complete parsed data
    """
    if not PARSED_DATA_DIR.exists():
        return set()
    
    processed = set()
    
    for site_dir in PARSED_DATA_DIR.iterdir():
        if not site_dir.is_dir():
            continue
        
        # Check if all required pickle files exist
        required_files = [
            'monthly_posts.pkl.gz',
            'tags.pkl.gz',
            'users.pkl.gz',
            'comments.pkl.gz'
        ]
        
        if all((site_dir / f).exists() for f in required_files):
            processed.add(site_dir.name)
    
    return processed


def determine_files_to_download(site_name: str, site_info: Dict) -> List[Tuple[str, str]]:
    """
    Determine which files to download based on site type.
    
    Args:
        site_name: Name of the site
        site_info: Site metadata from site_metadata.json
        
    Returns:
        List of tuples: [(filename, table_type), ...]
        table_type is 'complete_archive' or the table name (e.g., 'Posts')
    """
    files_to_download = []
    tables = site_info.get('tables', {})
    
    # Check if it has a complete_archive entry
    if 'complete_archive' in tables:
        filename = tables['complete_archive']['filename']
        files_to_download.append((filename, 'complete_archive'))
    
    else:
        # Individual table files (like Stack Overflow)
        required_tables = ['Posts', 'Tags', 'Users', 'Comments']
        
        for table in required_tables:
            if table in tables:
                filename = tables[table]['filename']
                files_to_download.append((filename, table))
            else:
                print(f"Warning: {table} not found for {site_name}")
    
    return files_to_download


def download_file(filename: str, dest_dir: Path) -> Path:
    """
    Download a file from Archive.org.
    
    Args:
        filename: Name of the file to download
        dest_dir: Destination directory
        
    Returns:
        Path to downloaded file
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        print(f"File already exists: {filename}")
        return output_path
    
    print(f"Downloading {filename}...")
    
    try:
        item = ia.get_item(STACKEXCHANGE_IDENTIFIER)
        file = item.get_file(filename)

        print(f"File size: {file.size / (1024*1024):.2f} MB")
        print(f"Downloading to: {output_path}")
        
        # Download with progress bar
        file.download(destdir=str(dest_dir))
        
        print(f"Downloaded {filename}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise


def extract_xml_files(archive_path: Path, extract_dir: Path, table_type: str) -> None:
    """
    Extract required XML files from .7z archive.
    
    Args:
        archive_path: Path to .7z file
        extract_dir: Directory to extract to
        table_type: 'complete_archive' or specific table name
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name}...")
    
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            if table_type == 'complete_archive':
                # Extract only required XMLs
                all_files = archive.getnames()
                files_to_extract = [f for f in all_files if f in REQUIRED_XMLS]
                
                if files_to_extract:
                    archive.extract(targets=files_to_extract, path=extract_dir)
                    print(f"Extracted {len(files_to_extract)} XML files")
                else:
                    print(f"No required XML files found in archive")
            
            else:
                # Extract single table XML (e.g., Posts.xml, Tags.xml)
                archive.extractall(path=extract_dir)
                print(f"Extracted {table_type}.xml")
                
    except Exception as e:
        print(f"Error extracting {archive_path.name}: {e}")
        raise


def parse_and_save(site_name: str, xml_dir: Path) -> None:
    """
    Parse XML files and save as compressed pickles.
    
    Args:
        site_name: Name of the site
        xml_dir: Directory containing XML files
    """
    output_dir = PARSED_DATA_DIR / site_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing and saving data for {site_name}...")
    
    # Parse Posts.xml
    posts_xml = xml_dir / "Posts.xml"
    if posts_xml.exists():
        monthly_posts = parse_posts_xml(str(posts_xml))
        with gzip.open(output_dir / "monthly_posts.pkl.gz", 'wb') as f:
            pickle.dump(monthly_posts, f)
        print(f"Saved monthly_posts.pkl.gz")
    else:
        raise FileNotFoundError(f"Posts.xml not found in {xml_dir}")
    
    # Parse Tags.xml
    tags_xml = xml_dir / "Tags.xml"
    if tags_xml.exists():
        tags = parse_tags_xml(str(tags_xml))
        with gzip.open(output_dir / "tags.pkl.gz", 'wb') as f:
            pickle.dump(tags, f)
        print(f"Saved tags.pkl.gz")
    else:
        raise FileNotFoundError(f"Tags.xml not found in {xml_dir}")
    
    # Parse Users.xml
    users_xml = xml_dir / "Users.xml"
    if users_xml.exists():
        users = parse_users_xml(str(users_xml))
        with gzip.open(output_dir / "users.pkl.gz", 'wb') as f:
            pickle.dump(users, f)
        print(f"Saved users.pkl.gz")
    else:
        raise FileNotFoundError(f"Users.xml not found in {xml_dir}")
    
    # Parse Comments.xml
    comments_xml = xml_dir / "Comments.xml"
    if comments_xml.exists():
        comments = parse_comments_xml(str(comments_xml))
        with gzip.open(output_dir / "comments.pkl.gz", 'wb') as f:
            pickle.dump(comments, f)
        print(f"Saved comments.pkl.gz")
    else:
        raise FileNotFoundError(f"Comments.xml not found in {xml_dir}")


def cleanup_raw_files(site_dir: Path) -> None:
    """
    Delete all .7z and .xml files and then remove the directory.
    
    Args:
        site_dir: Directory to clean up
    """
    print(f"Cleaning up raw files...")
    
    # Delete .7z files
    for f in site_dir.glob("*.7z"):
        f.unlink()
        print(f"Deleted {f.name}")
    
    # Delete .xml files
    for f in site_dir.glob("*.xml"):
        f.unlink()
        print(f"Deleted {f.name}")

    # Remove the empty directory
    if site_dir.exists() and not any(site_dir.iterdir()):
        site_dir.rmdir()
        print(f"Deleted directory {site_dir.name}")


def process_site(site_name: str, site_info: Dict) -> bool:
    """
    Process a single site: download, extract, parse, save, cleanup.
    
    Args:
        site_name: Name of the site
        site_info: Site metadata
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing: {site_name}")
    print(f"Size: {site_info['total_size_mb']:.2f} MB ({site_info['size_category']})")
    print(f"{'='*80}")
    
    site_raw_dir = RAW_DATA_DIR / site_name
    
    try:
        # Step 1: Determine files to download
        files_to_download = determine_files_to_download(site_name, site_info)
        print(f"Files to download: {len(files_to_download)}")
        
        # Step 2: Download files
        downloaded_files = []
        for filename, table_type in files_to_download:
            file_path = download_file(filename, site_raw_dir)
            downloaded_files.append((file_path, table_type))
        
        # Step 3: Extract XML files
        for archive_path, table_type in downloaded_files:
            extract_xml_files(archive_path, site_raw_dir, table_type)
        
        # Step 4: Parse and save
        parse_and_save(site_name, site_raw_dir)
        
        # Step 5: Cleanup
        cleanup_raw_files(site_raw_dir)
        
        print(f"Successfully processed {site_name}")
        return True
        
    except Exception as e:
        print(f"Error processing {site_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("="*80)
    print("Stack Exchange Download Pipeline")
    print("="*80)
    
    # Load metadata
    sites = load_site_metadata()
    print(f"Total sites in metadata: {len(sites)}")

    # Exclude Stack Overflow (too large, outlier)
    if 'stackoverflow.com' in sites:
        del sites['stackoverflow.com']
        print("Excluding stackoverflow.com")
    
    print(f"Sites to consider: {len(sites)}")
    
    # Check which sites are already processed
    processed_sites = get_processed_sites()
    print(f"Already processed: {len(processed_sites)}")
    
    # Determine sites to process
    sites_to_process = {name: info for name, info in sites.items() 
                        if name not in processed_sites}
    
    if not sites_to_process:
        print("\nAll sites already processed!")
        return
    
    print(f"Sites remaining: {len(sites_to_process)}")
    
    # Confirm before starting
    response = input("\nStart processing? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Process each site
    success_count = 0
    failed_sites = []
    
    for i, (site_name, site_info) in enumerate(sites_to_process.items(), 1):
        print(f"\n[{i}/{len(sites_to_process)}]")
        
        success = process_site(site_name, site_info)
        
        if success:
            success_count += 1
        else:
            failed_sites.append(site_name)
            print(f"\nFailed to process {site_name}. Stopping pipeline.")
            break
    
    # Summary
    print("\n" + "="*80)
    print("Pipeline Summary")
    print("="*80)
    print(f"Successfully processed: {success_count}/{len(sites_to_process)}")
    
    if failed_sites:
        print(f"\nFailed sites:")
        for site in failed_sites:
            print(f"  - {site}")
    else:
        print("\nAll sites processed successfully!")


if __name__ == "__main__":
    main()