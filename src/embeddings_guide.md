# Embeddings Guide

This guide explains the embeddings system used in the RAG-enhanced chatbot application.

## What are Embeddings?

Embeddings are vector representations of text that capture semantic meaning. In this application:

- Each chunk of data from your Snowflake database is converted to text
- Each text chunk is processed by OpenAI's embedding model `text-embedding-3-small` 
- The resulting vectors (1536 dimensions each) capture the meaning of the text
- These vectors allow for semantic search using cosine similarity

## Embeddings File

Your application stores embeddings in a file called `embeddings_store.pkl`. This file:

- Is approximately 3.20 MB in size
- Contains 214 embeddings (one for each chunk of data)
- Each embedding has 1536 dimensions
- Contains both the original text chunks and their vector representations

## Managing Embeddings

We've provided utilities to help you work with the embeddings:

### 1. View Embeddings
```bash
python view_embeddings.py
```
This shows basic information about your embeddings.

### 2. Manage Embeddings
```bash
python manage_embeddings.py
```
Without arguments, this shows detailed information about your embeddings file.

```bash
python manage_embeddings.py --delete
```
This deletes the embeddings file, forcing the application to regenerate embeddings on the next run.

### 3. When to Regenerate Embeddings

You should regenerate embeddings when:
- Your Snowflake data has changed significantly
- You modify the chunking strategy or embedding parameters
- The embeddings file is corrupted

To regenerate, simply delete the file and restart your application:
```bash
python manage_embeddings.py --delete
```

## Embedding Persistence System

The application now:
1. Checks if `embeddings_store.pkl` exists
2. If it exists, loads embeddings from the file (fast)
3. If not, generates new embeddings and saves them to disk (slower)
4. Shows clear status messages during the loading/generation process

This system ensures your application starts quickly after the first run.

## Technical Details

- **File format**: Python pickle (binary)
- **Model used**: text-embedding-3-small (OpenAI)
- **Dimensions**: 1536
- **Similarity metric**: Cosine similarity
- **Chunk types**:
  - Data chunks (200)
  - Schema information (1)
  - Statistical information (13)

## Troubleshooting

If you encounter issues with embeddings:

1. Delete the embeddings file to force regeneration
2. Check the OpenAI API key is valid
3. Verify you have appropriate permissions to write to the file
4. Ensure sufficient disk space is available