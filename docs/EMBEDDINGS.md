# Embeddings

The recommended semantic model is `BAAI/bge-small-en-v1.5`. Its model card reports 33.4 million
parameters, a 384-value output vector, and a 512-token sequence limit. It uses the MIT license:
[BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

ContextBench records the provider, model name, resolved immutable model revision, actual vector
dimension, and normalization setting in each index. Document indexing and query retrieval use the
provider's separate document and query encoders. Each index has a separate Qdrant collection.
Build a new index to change a model.

Sentence Transformers downloads models into the Hugging Face cache, usually under
`~/.cache/huggingface/hub`. ContextBench checks for cached model files first. If the model is not
present, the web app states the model name, approximate size, and cache location. The user must
approve the download with the explicit request field. That consent is cleared in the stored index
configuration after the build, so a later query cannot silently download the model again.

Install the optional runtime before using a semantic model:

```bash
uv sync --extra ml
```

The deterministic hash provider is for unit tests, offline demonstrations of pipeline mechanics,
and CI. Its results are labeled. They are not semantic embedding measurements.
