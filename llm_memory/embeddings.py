import uuid
import httpx
import numpy as np

from typing import Optional, List, Any
from data_types import EmbeddingConfig
from utils import is_valid_url
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core.node_parser import SentenceSplitter
from constants import MAX_EMBEDDING_DIM


def parse_and_chunk_text(text: str, chunk_size: int) -> List[str]:
    parser = SentenceSplitter(chunk_size=chunk_size)
    llama_index_docs = [LlamaIndexDocument(text=text)]
    nodes = parser.get_nodes_from_documents(llama_index_docs)
    return [n.text for n in nodes]


def query_embedding(embedding_model, query_text: str):
    """Generate padded embedding for querying database"""
    query_vec = embedding_model.get_text_embedding(query_text)
    query_vec = np.array(query_vec)
    query_vec = np.pad(
        query_vec, (0, MAX_EMBEDDING_DIM - query_vec.shape[0]), mode="constant"
    ).tolist()
    return query_vec


class EmbeddingEndpoint:
    """Implementation local embedding endpoint"""

    def __init__(
        self,
        model: str,
        base_url: str,
        user: str,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        if not is_valid_url(base_url):
            raise ValueError(
                f"Embeddings endpoint was provided an invalid URL (set to: '{base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )
        self.model_name = model
        self._user = user
        self._base_url = base_url
        self._timeout = timeout

    def _call_api(self, text: str) -> List[float]:
        if not is_valid_url(self._base_url):
            raise ValueError(
                f"Embeddings endpoint does not have a valid URL (set to: '{self._base_url}'). Make sure embedding_endpoint is set correctly in your MemGPT config."
            )

        headers = {"Content-Type": "application/json"}
        json_data = {"input": text, "model": self.model_name, "user": self._user}

        with httpx.Client() as client:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers=headers,
                json=json_data,
                timeout=self._timeout,
            )

        response_json = response.json()

        if isinstance(response_json, list):
            # embedding directly in response
            embedding = response_json
        elif isinstance(response_json, dict):
            # TEI embedding packaged inside openai-style response
            try:
                embedding = response_json["data"][0]["embedding"]
            except (KeyError, IndexError):
                raise TypeError(
                    f"Got back an unexpected payload from text embedding function, response=\n{response_json}"
                )
        else:
            # unknown response, can't parse
            raise TypeError(
                f"Got back an unexpected payload from text embedding function, response=\n{response_json}"
            )

        return embedding

    def get_text_embedding(self, text: str) -> List[float]:
        return self._call_api(text)


def embedding_model(config: EmbeddingConfig, user_id: Optional[uuid.UUID] = None):
    """Return LlamaIndex embedding model to use for embeddings"""

    return EmbeddingEndpoint(
        model=config.embedding_model,
        base_url=config.embedding_endpoint,
        user=user_id,
    )
