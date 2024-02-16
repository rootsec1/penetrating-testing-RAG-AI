import re
import chromadb

# Module imports
from PyPDF2 import PdfReader
from tqdm import tqdm

# Local
from constants import *
from util import *


def extract_text_from_pdf(pdf_path: str) -> str:
    complete_string = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            text = text.strip().lower()
            if text:
                complete_string += text + " "
    return complete_string


def clean_content(complete_string: str) -> list[str]:
    # Replace newlines with spaces
    sentences = complete_string.split(".")
    sentence_list = []
    for sentence in sentences:
        sentence = sentence.replace("\n", " ").strip()
        sentence = re.sub(r"\s+", " ", sentence)
        if sentence:
            sentence_list.append(sentence)
    return list(set(sentence_list))


def store_embeddings(documents: list[str], embeddings: list[str]):
    chromadb_client = chromadb.HttpClient(
        host=VECTOR_DB_HOST,
        port=VECTOR_DB_PORT
    )
    collection = chromadb_client.get_or_create_collection(
        name=VECTOR_DB_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("Hashing documents")
    id_list = [str(hash(document)) for document in documents]
    print("Storing documents in the collection batch-wise...")

    # Iterate through batches using BATCH_SIZE with tqdm progress bar
    docs_length = len(documents)

    for i in tqdm(range(0, len(documents), BATCH_SIZE)):
        print(
            f"Processing batch: { i // BATCH_SIZE + 1 } / { docs_length // BATCH_SIZE + 1 }"
        )
        collection.add(
            documents=documents[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            ids=id_list[i:i + BATCH_SIZE],
        )
    print("Documents stored successfully")


def main():
    # book_content = extract_text_from_epub("data/ceh_v12.epub")
    book_content = extract_text_from_pdf("data/wireshark_guide.pdf")
    sentences = clean_content(book_content)
    sentence_embeddings = encode_sentences(sentences)
    store_embeddings(sentences, sentence_embeddings)


if __name__ == '__main__':
    main()
