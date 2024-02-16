from util import *
from constants import *
from nltk.corpus import stopwords

import chromadb
import nltk
import streamlit as st
import replicate

# nltk.download('stopwords')
STOP_WORDS_SET = set(stopwords.words("english"))


def get_response_from_model(prompt: str) -> str:
    # The meta/llama-2-70b-chat model can stream output as it's running.
    # Input Dict
    input_dict = {
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": prompt,
        "temperature": 0.5,
        "system_prompt": "You are a cybersecurity expert with years of ethical hacking experience with a specialization in reverse engineering and networking. You kmow a lot of cool tools that can be used to exploit assets as you had used it during penetration testing.",
        "max_new_tokens": 1024,
        "min_new_tokens": -1
    }

    response_str = ""
    iterator = replicate.stream("meta/llama-2-70b-chat", input=input_dict)
    for event in iterator:
        response_str += event.data
    print(response_str)
    return response_str


def generate_prompt_with_rag(query: str) -> str:
    query = query.strip()
    chromadb_client = chromadb.HttpClient(host=VECTOR_DB_HOST)
    collection = chromadb_client.get_collection(name=VECTOR_DB_COLLECTION_NAME)
    # Remove stopwords from query
    query_filtered = " ".join(
        [word for word in query.split()
         if word not in STOP_WORDS_SET]
    )

    # Query the collection
    query_embedding = encode_sentences([query_filtered])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=20
    )
    documents = results.get("documents")
    documents = documents[0] if documents else documents
    documents = "\n\n".join(documents)
    prompt = RAG_PROMPT.format(documents, query)
    return prompt


def main():
    st.title("Wireshark Assistant")
    should_use_rag = st.select_slider(
        "Use RAG?:",
        ["No", "Yes"]
    )
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you with Wireshark?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if should_use_rag == "Yes":
            prompt = generate_prompt_with_rag(prompt)

        model_response = get_response_from_model(prompt)
        print(model_response)
        st.chat_message("assistant").markdown(model_response)


if __name__ == "__main__":
    main()
