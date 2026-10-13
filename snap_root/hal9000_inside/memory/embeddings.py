"""Local ONNX multilingual E5 embedding generator.

Enforces:
- 'query: ' prefix for search queries
- 'passage: ' prefix for stored memories
- L2-normalized 384-dimensional vectors
- Pure local execution via onnxruntime and tokenizers
"""

import json
import logging
import os
from typing import List, Optional, Union
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_E5_DIR = os.path.expanduser("~/.cache/hal9000/models/multilingual-e5-small")
E5_ONNX_PATH = os.path.join(DEFAULT_E5_DIR, "onnx/model.onnx")
E5_TOKENIZER_PATH = os.path.join(DEFAULT_E5_DIR, "onnx/tokenizer.json")
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class MultilingualE5Embedder:
    """Computes exact 384-dim normalized multilingual E5 embeddings."""

    def __init__(self, model_dir: str = DEFAULT_E5_DIR, providers: Optional[list] = None):
        self.model_dir = model_dir
        self.onnx_path = os.path.join(model_dir, "onnx/model.onnx")
        self.tokenizer_path = os.path.join(model_dir, "onnx/tokenizer.json")
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._tokenizer = None

    def load(self):
        """Loads ONNX session and tokenizer."""
        if not os.path.isfile(self.onnx_path) or not os.path.isfile(self.tokenizer_path):
            raise FileNotFoundError(
                f"E5 model assets not found in {self.model_dir}. Ensure model.onnx and tokenizer.json are downloaded."
            )
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        
        try:
            self._session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=self.providers)
        except Exception as e:
            logger.warning(f"Could not load E5 session with providers {self.providers} ({e}), falling back to CPU.")
            self._session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

        self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        # Enable truncation and padding to max length 512
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=None)
        logger.info(f"Loaded Multilingual E5 Embedder from {self.model_dir}")

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Applies mean pooling over token embeddings taking attention_mask into account."""
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors along last axis."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.clip(norms, a_min=1e-9, a_max=None)

    def encode(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Encode a list of texts into normalized 384-d vectors with correct E5 prefix."""
        if not self._session:
            self.load()

        prefix = "query: " if is_query else "passage: "
        prefixed_texts = [f"{prefix}{t.strip()}" for t in texts]

        # Tokenize batch
        encoded = self._tokenizer.encode_batch(prefixed_texts)
        input_ids = np.array([enc.ids for enc in encoded], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encoded], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Some ONNX models expect token_type_ids
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(None, inputs)
        # Last hidden state is usually output[0]
        token_embeddings = outputs[0]  # shape: (batch_size, seq_len, 384)
        
        pooled = self._mean_pooling(token_embeddings, attention_mask)
        normalized = self._normalize(pooled)
        return normalized.astype(float).tolist()

    def encode_query(self, query: str) -> List[float]:
        """Encode a single query with 'query: ' prefix."""
        return self.encode([query], is_query=True)[0]

    def encode_passage(self, passage: str) -> List[float]:
        """Encode a single passage with 'passage: ' prefix."""
        return self.encode([passage], is_query=False)[0]
