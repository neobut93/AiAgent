# Appendix: Manual embeddings vs `Chroma.from_documents()`

Short summary: `embeddings` are numeric vectors, not the original text. You can either compute embeddings yourself and call the Chroma client directly (manual), or hand documents + an embeddings object to LangChain's `Chroma.from_documents()` helper which computes and stores embeddings for you.

## Why the two patterns exist
- Manual (`embed_documents` + `collection.add`): you compute vectors explicitly, then push `ids`, `embeddings`, `documents`, and `metadatas` to Chroma. This gives you full control over batching, normalization, caching, and inspection of vectors.
- LangChain helper (`Chroma.from_documents`): convenience wrapper that accepts a list of `Document` objects and an embeddings object, computes embeddings internally, and persists everything. Quicker to prototype.

## Minimal examples

- Manual (explicit embeddings)

```python
# compute embeddings explicitly
texts = [d.page_content for d in docs]
vectors = embeddings.embed_documents(texts)
collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
```

- LangChain convenience (embeddings computed for you)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

emb = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, emb, persist_directory="./chroma_dir")
# `db` now contains docs, embeddings and metadata
```

## Pros / Cons
- Manual (more control)
  - Pros: full control over embedding calls, batching, normalization, vector caching, and reuse across collections.
  - Cons: more boilerplate.
- `from_documents` (convenience)
  - Pros: concise; LangChain handles embedding, chunk -> vector pipeline and persistence automatically.
  - Cons: less control and visibility into intermediate vectors; harder to intercept batching or custom normalization.

## When to use which
- Use `Chroma.from_documents` for fast prototyping and when you don't need custom embedding behaviour.
- Use explicit `embed_documents` + `collection.add` when you need to:
  - inspect or cache vectors, or reuse embeddings across stores
  - apply custom normalization or batching
  - precompute embeddings once and persist them separately

## Practical tip
If you expect to change embedding models often, compute and persist raw `page_content` (and metadata) alongside embeddings, and keep a small utility to re-embed and re-index when swapping models.
