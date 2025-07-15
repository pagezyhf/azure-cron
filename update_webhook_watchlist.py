#!/usr/bin/env python3
"""
Script to load a dataset with model_ids from the hub
and update the list of watched items of the webhook
"""

import sys
from datasets import load_dataset
from huggingface_hub import update_webhook, HfApi, get_webhook
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
    
    # Create watched items list from dataset rows
    watched_items = [{"type": row['type'], "name": row['id']} for row in dataset_train]
    logger.info(f"Found {len(watched_items)} unique watched items")
    logger.info(f"Watched items: {watched_items}")

    # Get current webhook configuration to preserve existing values
    try:
        current_webhook = get_webhook(webhook_id)
        
        # Update webhook preserving existing url, domains, and secret
        updated_webhook = update_webhook(
            webhook_id=webhook_id,
            watched=watched_items,
            url=current_webhook.url,
            domains=current_webhook.domains,
            secret=current_webhook.secret
        )
    except Exception as e:
        logger.error(f"Failed to get current webhook config: {e}")
    
    logger.info(f"Updated webhook: {updated_webhook}")

    return updated_webhook


def main():
    
    webhook_id = sys.argv[1]
    dataset_id = sys.argv[2]
    
    logger.info(f"Updating webhook: {webhook_id} with dataset: {dataset_id}")

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

    update_webhook_watched_items(dataset_id, webhook_id)

if __name__ == "__main__":
    main()
