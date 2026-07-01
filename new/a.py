from sentence_transformers import CrossEncoder

model_name = "BAAI/bge-reranker-v2-m3"

try:
    reranker = CrossEncoder(model_name)

    print("✅ Reranker loaded successfully")
    print("Model:", model_name)

    score = reranker.predict([
        (
            "What is retrieval augmented generation?",
            "Retrieval augmented generation combines retrieval systems with large language models."
        )
    ])

    print("Test score:", score[0])

except Exception as e:
    print("❌ Failed to load reranker")
    print(e)