from consts import TOP_K


def search_in_faiss(query, st_model, index, chunks, metadata, top_k=TOP_K):
    """Search the FAISS index for the most relevant chunks."""
    query_embedding = st_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append(
            {
                "chunk": chunks[idx].strip(),
                "source": metadata[idx].strip(),
                "distance": float(distances[0][i]),
            }
        )
    return results


def answer_with_model(query, st_model, index, chunks, metadata, answering_model):
    """Retrieve relevant chunks and generate an answer using LLM."""
    retrieved = search_in_faiss(query, st_model, index, chunks, metadata)
    context_text = "\n\n".join([f"From {r['source']}: {r['chunk']}" for r in retrieved])
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer only based on the given context."
    )
    response = answering_model.generate_content(prompt)
    return response.text.strip()
