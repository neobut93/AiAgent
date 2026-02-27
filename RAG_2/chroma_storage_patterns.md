# ChromaDB Storage Patterns

A reference guide for different ways to store and retrieve data in ChromaDB.

---

## 1. Default: LangChain `from_documents()` (Store Everything)

The simplest approach — stores `page_content`, `metadata`, and auto-computed embeddings.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chunks = [
    Document(page_content="The human heart pumps blood through the circulatory system."),
    Document(page_content="The circulatory system consists of heart, blood vessels, and blood."),
]

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_default"
)

# Query
results = db.similarity_search("heart function", k=3)
for doc in results:
    print(doc.page_content)
```

**What gets stored:**

| Field     | Value                                  |
| --------- | -------------------------------------- |
| Embedding | Auto-computed from `page_content`      |
| Document  | `page_content` string                  |
| Metadata  | `metadata` dict (empty `{}` if unset)  |

---

## 2. Custom Metadata on Documents

Add metadata to `Document` objects before storing. Useful for filtering during retrieval.

```python
chunks = [
    Document(
        page_content="The human heart pumps blood through the circulatory system.",
        metadata={"source": "textbook", "chapter": 3, "topic": "cardiology"}
    ),
    Document(
        page_content="The circulatory system consists of heart, blood vessels, and blood.",
        metadata={"source": "textbook", "chapter": 3, "topic": "circulation"}
    ),
]

db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_metadata")

# Query with metadata filter
results = db.similarity_search(
    "heart function",
    k=3,
    filter={"topic": "cardiology"}
)

# Query with multiple filters
results = db.similarity_search(
    "heart function",
    k=3,
    filter={"$and": [{"topic": "cardiology"}, {"chapter": 3}]}
)
```

**Supported filter operators:**

| Operator       | Example                                          |
| -------------- | ------------------------------------------------ |
| `$eq`          | `{"topic": {"$eq": "cardiology"}}`               |
| `$ne`          | `{"topic": {"$ne": "neurology"}}`                |
| `$gt`, `$gte`  | `{"chapter": {"$gt": 2}}`                        |
| `$lt`, `$lte`  | `{"chapter": {"$lt": 5}}`                        |
| `$in`          | `{"topic": {"$in": ["cardiology", "circ"]}}`     |
| `$and`, `$or`  | `{"$and": [filter1, filter2]}`                   |

---

## 3. Embed Full Text, Store Partial Text

Use the raw Chroma client to embed the full text but only store a truncated version.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_partial")
collection = client.get_or_create_collection("partial_text")

full_texts = [
    "The human heart is a muscular organ that pumps blood through the circulatory system. This blood delivers oxygen and nutrients to the body and removes metabolic waste.",
    "The average adult heart beats about 60 to 100 times per minute. The circulatory system consists of the heart, blood vessels, and blood."
]

# Embed the FULL text
vectors = embeddings.embed_documents(full_texts)

# Store TRUNCATED text with FULL-text embeddings
collection.add(
    ids=["doc1", "doc2"],
    embeddings=vectors,
    documents=[t[:60] + "..." for t in full_texts],
    metadatas=[
        {"full_text_length": len(t), "source": "textbook"}
        for t in full_texts
    ]
)

# Query
query_vector = embeddings.embed_query("heart function")
results = collection.query(query_embeddings=[query_vector], n_results=2)
print(results["documents"])
```

---

## 4. Embed Full Text, Store Summary

Embed the full chunk for accurate retrieval, but store a concise summary. Keep the full text in metadata.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_summary")
collection = client.get_or_create_collection("summaries")

full_texts = [
    "The human heart is a muscular organ that pumps blood through the circulatory system. This blood delivers oxygen and nutrients to the body and removes metabolic waste.",
    "The average adult heart beats about 60 to 100 times per minute. The circulatory system consists of the heart, blood vessels, and blood."
]

summaries = [
    "Heart pumps blood, delivers oxygen, removes waste.",
    "Heart beats 60-100 bpm; circulatory system overview."
]

# Embed FULL text
vectors = embeddings.embed_documents(full_texts)

# Store SUMMARY as document, FULL text in metadata
collection.add(
    ids=["doc1", "doc2"],
    embeddings=vectors,
    documents=summaries,
    metadatas=[
        {"full_text": ft, "source": "textbook"}
        for ft in full_texts
    ]
)

# Query — retrieval uses full-text embeddings, returns summary
query_vector = embeddings.embed_query("heart function")
results = collection.query(query_embeddings=[query_vector], n_results=2)

for i, doc in enumerate(results["documents"][0]):
    print(f"Summary: {doc}")
    print(f"Full text: {results['metadatas'][0][i]['full_text']}")
    print("---")
```

---

## 5. Store Only Embeddings + Metadata (No Document Text)

Store only embeddings and reference IDs. Manage text externally (e.g., SQL, S3).

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_refs_only")
collection = client.get_or_create_collection("refs_only")

full_texts = [
    "The human heart is a muscular organ that pumps blood...",
    "The average adult heart beats about 60 to 100 times...",
]

vectors = embeddings.embed_documents(full_texts)

# Store embeddings + metadata only (no documents param)
collection.add(
    ids=["doc1", "doc2"],
    embeddings=vectors,
    metadatas=[
        {"sql_id": 101, "table": "medical_docs"},
        {"sql_id": 102, "table": "medical_docs"},
    ]
)

# Query — get IDs/metadata, fetch full text from external store
query_vector = embeddings.embed_query("heart function")
results = collection.query(query_embeddings=[query_vector], n_results=2)

for meta in results["metadatas"][0]:
    print(f"Look up sql_id={meta['sql_id']} in table={meta['table']}")
```

---

## Quick Comparison

| Pattern          | Embed Source | Stored Text      | Use Case                        |
| ---------------- | ------------ | ---------------- | ------------------------------- |
| 1. Default       | page_content | page_content     | Simple RAG                      |
| 2. With metadata | page_content | page_content     | Filtered retrieval              |
| 3. Partial text  | Full text    | Truncated text   | Save storage space              |
| 4. Summary       | Full text    | Summary          | Better UX, save LLM tokens     |
| 5. No text       | Full text    | None (refs only) | External text storage (SQL/S3)  |

---

## 6. Inspecting & Debugging What's Stored

The Chroma Python object is a **thin client wrapper** — it doesn't hold documents as Python attributes. The actual text lives in the SQLite database on disk. That's why the VS Code debugger shows `_client`, `_collection`, `_collection_name` but **no visible text**.

To inspect the stored data, use the `.get()` method:

```python
# Get ALL stored data from a Chroma instance
data = vector_dbs[50].get()
print(data.keys())
# dict_keys(['ids', 'documents', 'metadatas', 'embeddings'])
```

Returns a dict like:

```python
{
    "ids":        ["abc123", "def456", ...],
    "documents":  ["The human heart is a...", "This blood delivers...", ...],
    "metadatas":  [{"source": "heart_document.txt"}, ...],
    "embeddings": None  # not returned by default (to save memory)
}
```

**Useful inspection methods:**

| Method | What it returns |
|---|---|
| `db.get()` | All docs, IDs, metadata |
| `db.get()["documents"]` | Just the text list |
| `db.get(include=["embeddings"])` | Docs + actual embedding vectors |
| `db._collection.count()` | Number of stored documents |
| `db.get(ids=["doc1"])` | Specific document by ID |
| `db.get(where={"source": "textbook"})` | Filter by metadata |

**Memory layout — why debugger shows no text:**

```
vector_dbs[50]  (Chroma object — what debugger shows)
    │
    ├── _client          → chromadb.api.client.Client  (connection)
    ├── _collection      → Collection(name=langchain)  (reference)
    ├── embeddings       → OpenAIEmbeddings(...)       (model)
    └── _collection_name → "langchain"
    
    ❌ No .documents attribute
    ❌ No .texts attribute

    Actual text lives in:
    ./chroma_db_size_50/chroma.sqlite3   ← on disk, accessed via .get()
```

---

## Key Takeaway


---

## 7. Chunking Strategies in LangChain

### CharacterTextSplitter

Splits text using a **single separator** (e.g., `\n`). Merges pieces up to `chunk_size`. If a line is longer than `chunk_size`, it cannot break it further.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)
chunks = splitter.split_text(text)
```

**Best for:** Simple, line-based splitting where lines are not too long.

---

### RecursiveCharacterTextSplitter

Splits text using a **priority list of separators**: `["\n\n", "\n", " ", ""]`. If a chunk is still too large, it falls back to the next separator, down to splitting by character.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_text(text)
```

**Best for:** General-purpose chunking, especially when text structure is inconsistent. Guarantees all chunks fit `chunk_size`.

---

### SemanticChunker

Splits text **by meaning** using embeddings. Steps:
1. Split text into sentences.
2. Embed each sentence.
3. Compare adjacent embeddings (cosine similarity).
4. Cut where similarity drops (semantic shift).

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings())
chunks = splitter.create_documents([text])
```

**Best for:** Keeping semantically related sentences together, e.g., for RAG or summarization.

---

### MarkdownHeaderTextSplitter

Splits markdown text at header levels (e.g., `#`, `##`, `###`).
For each chunk, attaches the **full header hierarchy** as metadata.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(text)
for doc in chunks:
    print(doc.page_content)
    print(doc.metadata)
```

**Best for:** Preserving document structure and enabling section-aware retrieval or filtering.

---

### Summary Table: Chunking Strategies

| Splitter                        | How it splits                | Handles long lines? | Preserves structure? | Best for                      |
|---------------------------------|------------------------------|---------------------|----------------------|-------------------------------|
| CharacterTextSplitter           | Single separator             | ❌                  | No                   | Simple, line-based text        |
| RecursiveCharacterTextSplitter  | Multiple fallback separators | ✅                  | No                   | General-purpose, robust        |
| SemanticChunker                 | Embedding similarity         | ✅                  | No                   | Semantic grouping              |
| MarkdownHeaderTextSplitter      | Markdown headers             | ✅                  | ✅                   | Section-aware chunking         |

---

## 8. Late Chunking (Attention Before Splitting)

**Late chunking** means embedding the full document first, then splitting the token embeddings into chunks. This leverages the transformer's self-attention: every token's embedding is influenced by the entire document before any chunking happens.

**How it works:**
- The full document is embedded, so each token's vector "knows" about the whole doc.
- After embedding, you split the token vectors into chunks and pool (e.g., average) them.
- Each chunk's vector carries both local meaning and global context.

**Analogy:**
> Like taking a group photo (full context) and then cropping out individuals—each crop still shows the relationships and context from the group.

**Limitation:**
- Only works up to the model's context window (e.g., 8,000 tokens). For longer docs, you must split first, so cross-section context is lost.

**Bottom line:**
Late chunking preserves more context per chunk because attention happens before splitting, not after.

---

## 9. Hypothetical Document Embedding (HyDE)

**HyDE** is a retrieval technique where, instead of embedding the user's query directly, you first generate a hypothetical answer to the query (using an LLM), then embed that answer and use it for retrieval.

**How it works:**
1. User query: "I want running shoes"
2. LLM generates a hypothetical answer (e.g., "Nike Air Jordan – Lightweight daily trainer...")
3. Embed the hypothetical answer
4. Use that embedding to search the vector database

**Why is this useful?**
- The hypothetical answer often contains more relevant terms and context than the original query, improving retrieval quality—especially for open-ended or underspecified queries.

**Example:**
```python
hyde_answer = "Nike Air Jordan – Lightweight daily trainer with responsive cushioning. ..."
hyde_query = query + " " + hyde_answer
hyde_embedding = embed(hyde_query)
results = db.similarity_search_by_vector(hyde_embedding)
```

**Bottom line:**
HyDE can boost retrieval performance by bridging the gap between user intent and document language, especially in semantic search and RAG pipelines.


---

## 10. Retrieval and Reranking Strategies

### Bi-encoder Retrieval

The bi-encoder encodes the query and each document independently into vectors using a dual-encoder model (e.g., SentenceTransformer). Retrieval is performed by computing the similarity (e.g., cosine similarity) between the query embedding and all document embeddings, and selecting the top-k most similar documents.

**Pros:** Fast, scalable to large collections, can precompute document embeddings.

**Cons:** Only captures surface-level similarity; may not distinguish subtle factual differences.

**Example:**
```python
query_emb = bi_encoder.encode(query, convert_to_tensor=True)
doc_embs = bi_encoder.encode(documents, convert_to_tensor=True)
cos_scores = util.cos_sim(query_emb, doc_embs)[0]
top_results = np.argsort(-cos_scores.cpu().numpy())[:100]
```

---

### Cross-encoder Reranking

The cross-encoder takes the query and each candidate document as a pair and jointly encodes them, producing a relevance score for each pair. This allows the model to reason about the factual relationship between the query and the document, often ranking the correct answer highest even among many similar candidates.

**Pros:** High accuracy, can distinguish subtle differences and factual correctness.

**Cons:** Slower and less scalable (must run for each query-document pair at inference time).

**Example:**
```python
pairs = [(query, doc) for doc in candidate_docs]
cross_scores = cross_encoder.predict(pairs)
reranked = sorted(zip(candidate_docs, cross_scores), key=lambda x: x[1], reverse=True)
```

---

### LLM-based Reranking (Listwise)

An LLM (e.g., GPT-4) is prompted with the query and a list of candidate documents, and asked to return the documents in order of relevance. The LLM uses its world knowledge and reasoning to identify the best answer, even among many similar candidates.

**Pros:** Can leverage deep factual and contextual knowledge; robust to paraphrasing and ambiguity.

**Cons:** Slowest and most expensive; not suitable for large candidate pools.

**Example:**
```python
prompt = f"You are a relevance ranking assistant. Query: {query}\nDocuments: ..."
response = client.chat.completions.create(...)
# Parse and reorder candidates based on LLM output
```

---

### Summary Table: Retrieval and Reranking

| Method                | How it works                | Speed    | Accuracy | Best for                        |
|-----------------------|-----------------------------|----------|----------|---------------------------------|
| Bi-encoder            | Vector similarity           | Fast     | Medium   | Large-scale retrieval           |
| Cross-encoder         | Joint query-doc scoring     | Medium   | High     | Top-k reranking, factual match  |
| LLM-based Reranking   | LLM ranks all candidates    | Slow     | Highest  | Small pools, complex reasoning  |

