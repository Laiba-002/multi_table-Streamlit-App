# """
# Utility script to view the contents of the embeddings pickle file.
# Run this script to see what's stored in the embeddings file.
# """
# import pickle
# import os
# import pandas as pd
# import numpy as np
# from pprint import pprint

# def load_vector_store(file_path="embeddings_store.pkl"):
#     """Load the vector store from disk."""
#     try:
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     except Exception as e:
#         print(f"Error loading embeddings: {str(e)}")
#         return None

# def view_embedding_stats(vector_store):
#     """Display statistics about the embeddings."""
#     if not vector_store:
#         return
    
#     embeddings = vector_store.get("embeddings", [])
#     chunks = vector_store.get("chunks", [])
    
#     print("\n===== EMBEDDING STATS =====")
#     print(f"Number of chunks: {len(chunks)}")
#     print(f"Number of embeddings: {len(embeddings)}")
    
#     if embeddings:
#         embedding_dimensions = len(embeddings[0])
#         print(f"Embedding dimensions: {embedding_dimensions}")
        
#         # Calculate average values and variance
#         if embedding_dimensions > 0:
#             embeddings_array = np.array(embeddings)
#             mean_values = np.mean(embeddings_array, axis=1)
#             print(f"Average embedding magnitude: {np.mean(np.linalg.norm(embeddings_array, axis=1)):.4f}")
#             print(f"Min embedding value: {np.min(embeddings_array):.4f}")
#             print(f"Max embedding value: {np.max(embeddings_array):.4f}")

# def view_chunk_samples(vector_store, num_samples=2):
#     """Display samples of the chunks."""
#     if not vector_store:
#         return
    
#     chunks = vector_store.get("chunks", [])
    
#     print("\n===== CHUNK SAMPLES =====")
#     if chunks:
#         for i, chunk in enumerate(chunks[:num_samples]):
#             print(f"\nChunk {i+1} (first 300 chars):")
#             print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
#     else:
#         print("No chunks found.")

# def view_embedding_samples(vector_store, num_samples=1):
#     """Display samples of the embeddings."""
#     if not vector_store:
#         return
    
#     embeddings = vector_store.get("embeddings", [])
    
#     print("\n===== EMBEDDING SAMPLES =====")
#     if embeddings:
#         for i, embedding in enumerate(embeddings[:num_samples]):
#             print(f"\nEmbedding {i+1} (first 20 dimensions):")
#             print(embedding[:20])
#             print(f"[... {len(embedding) - 20} more dimensions ...]")
#     else:
#         print("No embeddings found.")

# def main():
#     # Check if the embeddings file exists
#     file_path = "embeddings_store.pkl"
#     if not os.path.exists(file_path):
#         print(f"Error: The embeddings file '{file_path}' does not exist.")
#         return
    
#     # Load the vector store
#     print(f"Loading embeddings from {file_path}...")
#     vector_store = load_vector_store(file_path)
    
#     if not vector_store:
#         print("Failed to load vector store.")
#         return
    
#     print(f"\nSuccessfully loaded vector store!")
    
#     # Display the vector store structure
#     print("\n===== VECTOR STORE STRUCTURE =====")
#     print(f"Keys in vector store: {list(vector_store.keys())}")
    
#     # View stats, chunks, and embeddings
#     view_embedding_stats(vector_store)
#     view_chunk_samples(vector_store)
#     view_embedding_samples(vector_store)
    
#     print("\nDone!")

# if __name__ == "__main__":
#     main()




"""
Utility script to view the contents of the embeddings pickle file.
Run this script to see what's stored in the embeddings file.
"""
import pickle
import os
import pandas as pd
import numpy as np
from pprint import pprint

def load_vector_store(file_path="embeddings_store.pkl"):
    """Load the vector store from disk."""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None

def view_embedding_stats(vector_store):
    """Display statistics about the embeddings."""
    if not vector_store:
        return
    
    embeddings = vector_store.get("embeddings", [])
    chunks = vector_store.get("chunks", [])
    
    print("\n===== EMBEDDING STATS =====")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    
    if embeddings:
        embedding_dimensions = len(embeddings[0])
        print(f"Embedding dimensions: {embedding_dimensions}")
        
        # Calculate average values and variance
        if embedding_dimensions > 0:
            embeddings_array = np.array(embeddings)
            mean_values = np.mean(embeddings_array, axis=1)
            print(f"Average embedding magnitude: {np.mean(np.linalg.norm(embeddings_array, axis=1)):.4f}")
            print(f"Min embedding value: {np.min(embeddings_array):.4f}")
            print(f"Max embedding value: {np.max(embeddings_array):.4f}")

def view_chunk_samples(vector_store, num_samples=2):
    """Display samples of the chunks."""
    if not vector_store:
        return
    
    chunks = vector_store.get("chunks", [])
    
    print("\n===== CHUNK SAMPLES =====")
    if chunks:
        for i, chunk in enumerate(chunks[:num_samples]):
            print(f"\nChunk {i+1} (first 300 chars):")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    else:
        print("No chunks found.")

def view_embedding_samples(vector_store, num_samples=1):
    """Display samples of the embeddings."""
    if not vector_store:
        return
    
    embeddings = vector_store.get("embeddings", [])
    
    print("\n===== EMBEDDING SAMPLES =====")
    if embeddings:
        for i, embedding in enumerate(embeddings[:num_samples]):
            print(f"\nEmbedding {i+1} (first 20 dimensions):")
            print(embedding[:20])
            print(f"[... {len(embedding) - 20} more dimensions ...]")
    else:
        print("No embeddings found.")

def main():
    # Check if the embeddings file exists
    file_path = "embeddings_store.pkl"
    if not os.path.exists(file_path):
        print(f"Error: The embeddings file '{file_path}' does not exist.")
        return
    
    # Load the vector store
    print(f"Loading embeddings from {file_path}...")
    vector_store = load_vector_store(file_path)
    
    if not vector_store:
        print("Failed to load vector store.")
        return
    
    print(f"\nSuccessfully loaded vector store!")
    
    # Display the vector store structure
    print("\n===== VECTOR STORE STRUCTURE =====")
    print(f"Keys in vector store: {list(vector_store.keys())}")
    
    # View stats, chunks, and embeddings
    view_embedding_stats(vector_store)
    view_chunk_samples(vector_store)
    view_embedding_samples(vector_store)
    
    print("\nDone!")

if __name__ == "__main__":
    main()