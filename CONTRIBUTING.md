# Contributing

1. Use Python 3.11 and install `requirements-dev.txt`. Add `requirements.txt` only if you are working on the optional local Chroma indexer.
2. Keep provider keys in `.env` or the deployment secret store. Never commit credentials, uploaded documents, model artifacts, or local indexes.
3. Make ingestion limits explicit when adding a parser. Treat all uploaded bytes and extracted text as untrusted; do not execute uploaded content.
4. Add offline unit coverage for new parsing, retrieval, API, or security behavior. Tests must not download models or call external providers.
5. Run the commands in the README's Quality checks section before submitting changes.

For retrieval changes, report retrieval metrics against the golden set and include the evaluation configuration. Do not claim citation accuracy from the small existing sample as a general guarantee.
