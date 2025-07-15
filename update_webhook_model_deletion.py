#!/usr/bin/env python3
"""
Script to load the private dataset pagezyhf/azure-models from Hugging Face Hub
and update webhook watched items with the models in the dataset
"""

import os
from datasets import load_dataset
from huggingface_hub import update_webhook, HfApi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_webhook_watched_items(dataset_id, webhook_id):
    """
    Update webhook watched items with models from the dataset
    """

    dataset = load_dataset(dataset_id)
    dataset_train = dataset['train']
    
    # Extract unique model names
    model_names = []
    model_names = list({row['model_id'] for row in dataset_train if row['model_id']})
    logger.info(f"Found {len(model_names)} unique models")
    
    # Create watched items list for models
    watched_items = []
    for model_name in model_names:
        watched_items.append({
            "type": "model",
            "name": model_name
        })

    # update webhook    
    updated_webhook = update_webhook(
        webhook_id=webhook_id,
        watched=watched_items
    )
    logger.info(f"Updated webhook: {updated_webhook}")
    
    return updated_webhook


def main():
    # Configuration
    WEBHOOK_ID = "686fc4680e57742d7a789d41"
    DATASET_ID = "pagezyhf/azure-models"

    # Check Hugging Face Hub login status
    hf_api_for_login_check = HfApi()
    try:
        user_info = hf_api_for_login_check.whoami()
        logger.info(f"Successfully logged in to Hugging Face Hub as {user_info['name']}.")
    except Exception:  # Catches HTTPError (e.g., 401) if not logged in, or other network issues.
        logger.error(
            "Failed to verify Hugging Face Hub login status. "
            "Please ensure you are logged in using 'huggingface-cli login'. "
            "The script needs to push data to the Hub."
        )
        print("Exiting due to authentication issue. Please run 'huggingface-cli login'.")
        return # Exit the main function if not logged in.

    update_webhook_watched_items(DATASET_ID, WEBHOOK_ID)

if __name__ == "__main__":
    main()
