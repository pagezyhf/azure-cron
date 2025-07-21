# Update Webhook Watchlist

Automatically updates Hugging Face webhook watchlists daily, based on model lists in HF datasets.

- Loads model IDs from HF datasets (`hf-azure-internal/model-catalog` and `hf-azure-internal/simship-orgs`)
- Updates webhook watchlists to track those models (webhook are owned by https://huggingface.co/pagezyhf)
- Runs daily at 12:00 UTC via GitHub Actions

Requires `HF_TOKEN_UPDATE_WEBHOOK_WATCHLIST` secret to be set at github repo level for authentication, with read access to `hf-azure-internal`repo and write access to `pagezyhf` webhooks.

# Trending Models Analysis

Creates a daily report on top 200 trending models and their support for the Azure Model Catalog.

- load top 200 trending models
- check if each model fits the prerequisites to be added to the HF collection in the Azure Model Catalog
- save to a dataset in `hf-azure-internal`repo, used by a HF Space for visualizations

Requires `HF_TOKEN_TRENDING_MODELS_ANALYSIS` secret to be set at github repo level for authentication, with write access to `hf-azure-internal`repo.
