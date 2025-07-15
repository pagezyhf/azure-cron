Automatically updates Hugging Face webhook watchlists daily, based on model lists in HF datasets.

- Loads model IDs from HF datasets (`hf-azure-internal/model-catalog` and `hf-azure-internal/simship-orgs`)
- Updates webhook watchlists to track those models (webhook are owned by https://huggingface.co/pagezyhf)
- Runs daily at 12:00 UTC via GitHub Actions

Requires `HF_TOKEN` secret to be set at github repo level for authentication.