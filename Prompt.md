# Hybrid Semantic Search System Implementation Guide

## Overview

Build a modular hybrid search system that combines semantic search (using embeddings) with traditional keyword search (TF-IDF) for enhanced retrieval performance. The system uses FAISS for vector storage and is designed to be FastAPI-ready.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Input    │    │  Data Pipeline  │    │  Search Engine  │
│                 │    │                 │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Documents     │───▶│ • Text Cleaning │───▶│ • Semantic      │
│ • Queries       │    │ • Chunking      │    │ • Keyword       │
│ • Metadata      │    │ • Embeddings    │    │ • Hybrid Fusion │
└─────────────────┘    │ • TF-IDF        │    │ • Ranking       │
                       │ • FAISS Index   │    └─────────────────┘
                       └─────────────────┘
```

## Core Requirements

### Dependencies
```bash
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install scikit-learn
pip install numpy pandas
pip install nltk
pip install fastapi uvicorn  # for API wrapper
```

## Module Structure

Create the following modular structure in VS Code:

```
hybrid_search/
├── __init__.py
├── config.py              # Configuration settings
├── text_processor.py      # Text cleaning and preprocessing
├── chunking.py           # Document chunking strategies
├── embeddings.py         # Embedding generation (local)
├── keyword_search.py     # TF-IDF implementation
├── vector_store.py       # FAISS operations
├── hybrid_engine.py      # Hybrid search logic
├── ranking.py            # Result fusion and ranking
├── utils.py              # Utility functions
├── tests/                # Unit tests for each module
│   ├── test_text_processor.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_keyword_search.py
│   ├── test_vector_store.py
│   ├── test_hybrid_engine.py
│   └── test_ranking.py
└── api/                  # FastAPI wrapper (future)
    ├── __init__.py
    ├── main.py
    ├── models.py
    └── endpoints.py
```

## Implementation Requirements

### 1. Configuration Module (`config.py`)
- Default embedding model settings
- FAISS index configurations
- TF-IDF parameters
- Chunking strategies
- Search fusion weights

### 2. Text Processor (`text_processor.py`)
**Functions to implement:**
- `clean_text(text: str) -> str`
- `normalize_text(text: str) -> str`
- `remove_stopwords(text: str, custom_stopwords: List[str] = None) -> str`
- `preprocess_for_embedding(text: str) -> str`
- `preprocess_for_tfidf(text: str) -> str`

**Requirements:**
- Handle multiple languages if needed
- Preserve sentence boundaries
- Remove noise while maintaining semantic meaning
- Different preprocessing for embeddings vs TF-IDF

### 3. Document Chunking (`chunking.py`)
**Functions to implement:**
- `chunk_by_sentences(text: str, max_sentences: int, overlap: int) -> List[str]`
- `chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]`
- `chunk_by_paragraphs(text: str, max_paragraphs: int) -> List[str]`
- `smart_chunk(text: str, strategy: str, **kwargs) -> List[Dict]`

**Requirements:**
- Maintain context across chunks
- Preserve metadata for each chunk
- Support multiple chunking strategies
- Handle edge cases (very short/long texts)

### 4. Embeddings Module (`embeddings.py`)
**Functions to implement:**
- `load_embedding_model(model_name: str) -> SentenceTransformer`
- `generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray`
- `batch_encode(texts: List[str], batch_size: int, model: SentenceTransformer) -> np.ndarray`
- `normalize_embeddings(embeddings: np.ndarray) -> np.ndarray`

**Requirements:**
- Support multiple embedding models
- Batch processing for large datasets
- Memory-efficient processing
- Normalization for cosine similarity

### 5. Keyword Search (`keyword_search.py`)
**Functions to implement:**
- `create_tfidf_vectorizer(**kwargs) -> TfidfVectorizer`
- `fit_tfidf(texts: List[str], vectorizer: TfidfVectorizer) -> sparse.csr_matrix`
- `search_tfidf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix: sparse.csr_matrix, top_k: int) -> List[Dict]`
- `get_feature_names(vectorizer: TfidfVectorizer) -> List[str]`

**Requirements:**
- Configurable TF-IDF parameters
- Support for custom vocabulary
- Efficient sparse matrix operations
- Query expansion capabilities

### 6. Vector Store (`vector_store.py`)
**Functions to implement:**
- `create_faiss_index(embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index`
- `add_to_index(index: faiss.Index, embeddings: np.ndarray) -> None`
- `search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]`
- `save_faiss_index(index: faiss.Index, path: str) -> None`
- `load_faiss_index(path: str) -> faiss.Index`

**Index Types to Support:**
- `IndexFlatIP`: Exact search with inner product
- `IndexFlatL2`: Exact search with L2 distance
- `IndexIVFFlat`: Faster approximate search
- `IndexHNSW`: Hierarchical navigable small world

### 7. Hybrid Engine (`hybrid_engine.py`)
**Main class: `HybridSearchEngine`**

**Methods to implement:**
- `__init__(config: Dict)`
- `prepare_corpus(documents: List[Dict]) -> None`
- `semantic_search(query: str, top_k: int) -> List[Dict]`
- `keyword_search(query: str, top_k: int) -> List[Dict]`
- `hybrid_search(query: str, top_k: int, semantic_weight: float, keyword_weight: float) -> List[Dict]`
- `add_documents(new_documents: List[Dict]) -> None`
- `save_index(path: str) -> None`
- `load_index(path: str) -> None`

### 8. Result Fusion and Ranking (`ranking.py`)
**Functions to implement:**
- `rrf_fusion(semantic_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]`
- `weighted_fusion(semantic_results: List[Dict], keyword_results: List[Dict], semantic_weight: float, keyword_weight: float) -> List[Dict]`
- `normalize_scores(results: List[Dict]) -> List[Dict]`
- `rerank_results(results: List[Dict], query: str, rerank_model: str = None) -> List[Dict]`

**Fusion Strategies:**
- Reciprocal Rank Fusion (RRF)
- Weighted score combination
- Borda count
- CombSUM/CombMNZ

### 9. Utilities (`utils.py`)
**Functions to implement:**
- `load_documents_from_directory(path: str) -> List[Dict]`
- `save_results_to_json(results: List[Dict], path: str) -> None`
- `calculate_metrics(ground_truth: List[Dict], results: List[Dict]) -> Dict`
- `benchmark_search_performance(engine: HybridSearchEngine, queries: List[str]) -> Dict`

## Testing Strategy

### Unit Tests for Each Module

**`test_text_processor.py`**
```python
def test_clean_text():
    # Test basic cleaning
    # Test edge cases (empty, very long text)
    # Test special characters handling

def test_normalize_text():
    # Test Unicode normalization
    # Test case handling
    # Test accent removal

def test_stopwords_removal():
    # Test default stopwords
    # Test custom stopwords
    # Test language-specific stopwords
```

**`test_embeddings.py`**
```python
def test_model_loading():
    # Test different model names
    # Test invalid models
    # Test model caching

def test_embedding_generation():
    # Test single text
    # Test batch processing
    # Test empty inputs
    # Test very long texts

def test_embedding_normalization():
    # Test L2 normalization
    # Test cosine similarity preservation
```

**`test_vector_store.py`**
```python
def test_faiss_index_creation():
    # Test different index types
    # Test index with different dimensions
    # Test empty embeddings

def test_faiss_search():
    # Test exact search
    # Test approximate search
    # Test edge cases (k > index size)

def test_index_persistence():
    # Test save/load functionality
    # Test index integrity after loading
```

**Integration Tests**
- End-to-end pipeline testing
- Performance benchmarking
- Memory usage monitoring
- Search quality evaluation

## Data Structures

### Document Format
```python
{
    "id": "unique_document_id",
    "content": "Full text content of the document",
    "title": "Document title",
    "source": "source_file.pdf",
    "category": "document_category",
    "metadata": {
        "author": "Author name",
        "date": "2024-01-01",
        "tags": ["tag1", "tag2"],
        "custom_field": "custom_value"
    }
}
```

### Chunk Format
```python
{
    "chunk_id": "doc_id_chunk_0",
    "parent_doc_id": "unique_document_id",
    "text": "Chunk text content",
    "start_char": 0,
    "end_char": 500,
    "metadata": {
        # Inherited from parent document
        # Plus chunk-specific metadata
        "chunk_index": 0,
        "chunk_size": 500,
        "overlap_size": 50
    }
}
```

### Search Result Format
```python
{
    "chunk_id": "doc_id_chunk_0",
    "text": "Retrieved text",
    "score": 0.85,
    "search_type": "semantic|keyword|hybrid",
    "metadata": {
        # Document and chunk metadata
    },
    "highlights": ["highlighted", "terms"],  # For keyword search
    "rank": 1
}
```

## Performance Requirements

### Indexing Performance
- Process 10K documents in < 5 minutes
- Memory usage < 4GB for 100K documents
- Incremental indexing capability

### Search Performance
- Query response time < 100ms for 100K documents
- Support for concurrent queries
- Batch query processing

### Quality Metrics to Track
- Precision@K (K=1,5,10)
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

## Configuration Example

```python
# config.py
SEARCH_CONFIG = {
    "embedding": {
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "normalize": True
    },
    "chunking": {
        "strategy": "sentences",
        "chunk_size": 3,
        "overlap": 1,
        "min_chunk_size": 50
    },
    "tfidf": {
        "max_features": 10000,
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": (1, 2),
        "stop_words": "english"
    },
    "faiss": {
        "index_type": "IndexFlatIP",
        "nlist": 100,  # for IVF indices
        "m": 8,        # for PQ indices
        "nbits": 8     # for PQ indices
    },
    "hybrid": {
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "fusion_method": "rrf",
        "rrf_k": 60
    },
    "search": {
        "default_top_k": 10,
        "max_top_k": 100,
        "min_score_threshold": 0.1
    }
}
```

## Usage Examples

### Basic Usage
```python
# Initialize search engine
from hybrid_search import HybridSearchEngine

engine = HybridSearchEngine()

# Prepare documents
documents = [
    {
        "content": "Your document content here...",
        "title": "Document Title",
        "source": "source.pdf"
    }
]

engine.prepare_corpus(documents)

# Search
results = engine.hybrid_search("your query", top_k=10)
```

### Advanced Usage
```python
# Custom configuration
config = {
    "semantic_weight": 0.8,
    "keyword_weight": 0.2,
    "fusion_method": "weighted"
}

engine = HybridSearchEngine(config)

# Different search modes
semantic_results = engine.semantic_search("query", top_k=5)
keyword_results = engine.keyword_search("query", top_k=5)
hybrid_results = engine.hybrid_search("query", top_k=5)

# Add new documents incrementally
new_docs = [{"content": "New document...", "title": "New Title"}]
engine.add_documents(new_docs)

# Save and load
engine.save_index("./search_index")
engine.load_index("./search_index")
```

## Future FastAPI Integration

The modular design allows easy FastAPI wrapper:

```python
# api/main.py
from fastapi import FastAPI
from hybrid_search import HybridSearchEngine

app = FastAPI()
search_engine = HybridSearchEngine()

@app.post("/search/")
async def search(query: str, top_k: int = 10):
    results = search_engine.hybrid_search(query, top_k)
    return {"query": query, "results": results}

@app.post("/documents/")
async def add_documents(documents: List[Dict]):
    search_engine.add_documents(documents)
    return {"status": "success", "added": len(documents)}
```

## Development Workflow

1. **Setup Environment**: Create virtual environment and install dependencies
2. **Implement Core Modules**: Start with `text_processor.py`, then `chunking.py`, etc.
3. **Write Unit Tests**: Test each function independently
4. **Integration Testing**: Test full pipeline
5. **Performance Testing**: Benchmark with realistic datasets
6. **API Wrapper**: Implement FastAPI endpoints
7. **Deployment**: Containerize and deploy

## Quality Assurance

### Code Quality
- Use type hints throughout
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Implement logging for debugging

### Testing Coverage
- Aim for >90% test coverage
- Include edge case testing
- Performance regression testing
- Memory leak detection

### Documentation
- API documentation with examples
- Performance benchmarks
- Troubleshooting guide
- Configuration reference

This modular approach ensures each component can be developed, tested, and maintained independently while supporting easy integration into a FastAPI service.