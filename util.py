import torch

from sentence_transformers import SentenceTransformer
from constants import *


def convert_tensor_to_float_list(tensor):
    tensor_list = tensor.tolist()
    # Convert tensor to list of floats and expand inner list if exists
    return tensor_list


def encode_sentences(sentences: list[str]) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE USED: {device}")
    sentence_encoder = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL
    ).to(device)
    print("Encoding sentences of length: ", len(sentences))
    sentence_embeddings = sentence_encoder.encode(
        sentences,
        convert_to_tensor=True
    ).to(device)
    return convert_tensor_to_float_list(sentence_embeddings)
