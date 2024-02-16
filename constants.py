VECTOR_DB_HOST = "localhost"
VECTOR_DB_PORT = 8000
VECTOR_DB_COLLECTION_NAME = "cybersec"

BATCH_SIZE = 1000
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
DISTANCE_THRESHOLD = 0.3

RAG_PROMPT = """
Use the following content as context to give me more precise and concise steps for my question.
{}

Question: {}
"""