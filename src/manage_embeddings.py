# """
# Utility script to manage the embeddings store pickle file.
# This script allows you to:
# 1. Check if embeddings need to be regenerated based on data changes
# 2. Delete the embeddings file to force regeneration
# 3. View detailed information about the stored embeddings
# """
# import pickle
# import os
# import argparse
# import pandas as pd
# import numpy as np
# from datetime import datetime

# def load_vector_store(file_path="embeddings_store.pkl"):
#     """Load the vector store from disk."""
#     try:
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     except Exception as e:
#         print(f"Error loading embeddings: {str(e)}")
#         return None

# def delete_embeddings_file(file_path="embeddings_store.pkl"):
#     """Delete the embeddings file to force regeneration."""
#     if os.path.exists(file_path):
#         try:
#             os.remove(file_path)
#             print(f"Successfully deleted {file_path}. Embeddings will be regenerated on next app start.")
#             return True
#         except Exception as e:
#             print(f"Error deleting embeddings file: {str(e)}")
#             return False
#     else:
#         print(f"No embeddings file found at {file_path}.")
#         return False

# def get_file_stats(file_path="embeddings_store.pkl"):
#     """Get file statistics for the embeddings file."""
#     if not os.path.exists(file_path):
#         print(f"No embeddings file found at {file_path}.")
#         return None
    
#     stats = {}
#     stats["file_size_mb"] = os.path.getsize(file_path) / (1024 * 1024)
#     stats["created"] = datetime.fromtimestamp(os.path.getctime(file_path))
#     stats["modified"] = datetime.fromtimestamp(os.path.getmtime(file_path))
    
#     return stats

# def view_detailed_info(file_path="embeddings_store.pkl"):
#     """View detailed information about the embeddings file."""
#     # Get file stats
#     stats = get_file_stats(file_path)
#     if not stats:
#         return
    
#     print("\n===== EMBEDDINGS FILE INFO =====")
#     print(f"File path: {os.path.abspath(file_path)}")
#     print(f"File size: {stats['file_size_mb']:.2f} MB")
#     print(f"Created: {stats['created']}")
#     print(f"Last modified: {stats['modified']}")
    
#     # Load and analyze vector store
#     vector_store = load_vector_store(file_path)
#     if not vector_store:
#         return
    
#     embeddings = vector_store.get("embeddings", [])
#     chunks = vector_store.get("chunks", [])
    
#     print("\n===== VECTOR STORE CONTENT =====")
#     print(f"Number of chunks: {len(chunks)}")
#     print(f"Number of embeddings: {len(embeddings)}")
    
#     if embeddings:
#         embedding_dimensions = len(embeddings[0])
#         print(f"Embedding dimensions: {embedding_dimensions}")
        
#         # Show memory usage estimate
#         embeddings_size_mb = (len(embeddings) * embedding_dimensions * 4) / (1024 * 1024)  # Assuming float32
#         print(f"Estimated embeddings memory usage: {embeddings_size_mb:.2f} MB")
    
#     # Show a few chunk samples
#     if chunks:
#         print("\n----- Chunk Types -----")
#         chunk_types = {}
#         for chunk in chunks:
#             # Identify chunk type by looking at the first line
#             first_line = chunk.split('\n')[0] if '\n' in chunk else chunk[:50]
#             if first_line not in chunk_types:
#                 chunk_types[first_line] = 0
#             chunk_types[first_line] += 1
        
#         for chunk_type, count in chunk_types.items():
#             print(f"{chunk_type}: {count} chunks")

# def main():
#     parser = argparse.ArgumentParser(description="Manage embeddings store")
#     parser.add_argument("--view", action="store_true", help="View detailed information about the embeddings")
#     parser.add_argument("--delete", action="store_true", help="Delete the embeddings file to force regeneration")
    
#     args = parser.parse_args()
    
#     # If no arguments provided, show help
#     if not (args.view or args.delete):
#         parser.print_help()
#         view_detailed_info()  # Default action is to view info
#         return
    
#     # Process arguments
#     if args.delete:
#         delete_embeddings_file()
    
#     if args.view:
#         view_detailed_info()

# if __name__ == "__main__":
#     main()




"""
Utility script to manage the embeddings store pickle file.
This script allows you to:
1. Check if embeddings need to be regenerated based on data changes
2. Delete the embeddings file to force regeneration
3. View detailed information about the stored embeddings
"""
import pickle
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

def load_vector_store(file_path="embeddings_store.pkl"):
    """Load the vector store from disk."""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None

def delete_embeddings_file(file_path="embeddings_store.pkl", metadata_path="embeddings_metadata.json"):
    """Delete the embeddings file to force regeneration."""
    deleted_files = []
    
    # Delete the embeddings file
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except Exception as e:
            print(f"Error deleting embeddings file: {str(e)}")
    
    # Delete the metadata file
    if os.path.exists(metadata_path):
        try:
            os.remove(metadata_path)
            deleted_files.append(metadata_path)
        except Exception as e:
            print(f"Error deleting metadata file: {str(e)}")
    
    if deleted_files:
        print(f"Successfully deleted: {', '.join(deleted_files)}")
        print("Embeddings will be regenerated on next app start.")
        return True
    else:
        print("No embeddings or metadata files found.")
        return False

def get_file_stats(file_path="embeddings_store.pkl"):
    """Get file statistics for the embeddings file."""
    if not os.path.exists(file_path):
        print(f"No embeddings file found at {file_path}.")
        return None
    
    stats = {}
    stats["file_size_mb"] = os.path.getsize(file_path) / (1024 * 1024)
    stats["created"] = datetime.fromtimestamp(os.path.getctime(file_path))
    stats["modified"] = datetime.fromtimestamp(os.path.getmtime(file_path))
    
    return stats

def view_detailed_info(file_path="embeddings_store.pkl", metadata_path="embeddings_metadata.json"):
    """View detailed information about the embeddings file."""
    # Get file stats
    stats = get_file_stats(file_path)
    if not stats:
        return
    
    print("\n===== EMBEDDINGS FILE INFO =====")
    print(f"File path: {os.path.abspath(file_path)}")
    print(f"File size: {stats['file_size_mb']:.2f} MB")
    print(f"Created: {stats['created']}")
    print(f"Last modified: {stats['modified']}")
    
    # Check for metadata file
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
            
            print("\n===== METADATA INFO =====")
            print(f"Table name: {metadata.get('table_name', 'Unknown')}")
            print(f"Created at: {metadata.get('created_at', 'Unknown')}")
            print(f"Number of chunks: {metadata.get('num_chunks', 'Unknown')}")
            print(f"Number of embeddings: {metadata.get('num_embeddings', 'Unknown')}")
            
            # Display columns
            if 'columns' in metadata:
                print(f"\nColumns ({len(metadata['columns'])}):")
                for col in metadata['columns']:
                    print(f"  - {col}")
        except Exception as e:
            print(f"\nError reading metadata file: {str(e)}")
    
    # Load and analyze vector store
    vector_store = load_vector_store(file_path)
    if not vector_store:
        return
    
    embeddings = vector_store.get("embeddings", [])
    chunks = vector_store.get("chunks", [])
    
    print("\n===== VECTOR STORE CONTENT =====")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    
    if embeddings:
        embedding_dimensions = len(embeddings[0])
        print(f"Embedding dimensions: {embedding_dimensions}")
        
        # Show memory usage estimate
        embeddings_size_mb = (len(embeddings) * embedding_dimensions * 4) / (1024 * 1024)  # Assuming float32
        print(f"Estimated embeddings memory usage: {embeddings_size_mb:.2f} MB")
    
    # Show a few chunk samples
    if chunks:
        print("\n----- Chunk Types -----")
        chunk_types = {}
        for chunk in chunks:
            # Identify chunk type by looking at the first line
            first_line = chunk.split('\n')[0] if '\n' in chunk else chunk[:50]
            if first_line not in chunk_types:
                chunk_types[first_line] = 0
            chunk_types[first_line] += 1
        
        for chunk_type, count in chunk_types.items():
            print(f"{chunk_type}: {count} chunks")

def main():
    parser = argparse.ArgumentParser(description="Manage embeddings store")
    parser.add_argument("--view", action="store_true", help="View detailed information about the embeddings")
    parser.add_argument("--delete", action="store_true", help="Delete the embeddings file to force regeneration")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.view or args.delete):
        parser.print_help()
        view_detailed_info()  # Default action is to view info
        return
    
    # Process arguments
    if args.delete:
        delete_embeddings_file()
    
    if args.view:
        view_detailed_info()

if __name__ == "__main__":
    main()