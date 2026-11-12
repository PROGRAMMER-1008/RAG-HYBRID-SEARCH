import numpy as np
import pytest

from app.retrieval.embedding import FastEmbedEncoder


class FakeFastEmbedModel:
    def embed(self, texts, *, batch_size, parallel):
        assert batch_size == 8
        assert parallel is None
        for text in texts:
            if text == "bad-shape":
                yield np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            else:
                yield np.asarray([3.0, 4.0], dtype=np.float32)


def test_fastembed_adapter_returns_normalized_float_vectors():
    encoder = FastEmbedEncoder("test-model", "/tmp/models", model=FakeFastEmbedModel())

    vectors = encoder.encode(["first", "second"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_fastembed_adapter_rejects_invalid_output_shape():
    encoder = FastEmbedEncoder("test-model", "/tmp/models", model=FakeFastEmbedModel())

    with pytest.raises(RuntimeError, match="invalid vector matrix"):
        encoder.encode(["bad-shape"])
