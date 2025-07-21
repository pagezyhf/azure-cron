from huggingface_hub import HfApi, get_repo_discussions
from datasets import Dataset, concatenate_datasets
from datetime import datetime
import pandas as pd
import sys
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# Constants
COLLECTION_DATE = datetime.utcnow().isoformat()
EXCLUDED_ORGS = ['meta-llama', 'mistralai']
ALLOWED_LICENSES = [
    "apache-2.0",
    "mit",
    "creativeml-openrail-m",
    "openrail",
    "afl-3.0",
    "cc-by-4.0",
    "other",
    "cc-by-sa-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-4.0",
    "gpl-3.0",
    "artistic-2.0",
    "cc",
    "bigscience-bloom-rail-1.0",
    "bsd-3-clause",
    "cc0-1.0",
    "bigscience-openrail-m",
    "wtfpl",
    "agpl-3.0",
    "unlicense",
    "gpl",
    "bsl-1.0",
    "openrail++",
    "bsd",
    "cc-by-sa-3.0",
    "cc-by-3.0",
    "bsd-2-clause",
    "gpl-2.0",
    "cc-by-2.0",
    "cc-by-nc-3.0",
    "lgpl-3.0",
    "cc-by-nc-nd-4.0",
    "osl-3.0",
    "pddl",
    "cc-by-nc-sa-3.0",
    "bsd-3-clause-clear",
    "lgpl",
    "lgpl-lr",
    "ecl-2.0",
    "gfdl",
    "cc-by-nd-4.0",
    "c-uda",
    "mpl-2.0",
    "isc",
    "odc-by",
    "ms-pl",
    "zlib",
    "odbl",
    "cc-by-nc-sa-2.0",
    "ncsa",
    "cc-by-nc-2.0",
    "cc-by-2.5",
    "ofl-1.1",
    "epl-1.0",
    "cc-by-nc-nd-3.0",
    "eupl-1.1",
]
SUPPORTED_TASKS = [
    "feature-extraction",
    "automatic-speech-recognition",
    "text-to-speech",
    "speech-to-text",
    "translation",
    "text-translation",
    "question-answering",
    "text-classification",
    "fill-mask",
    "token-classification",
    "text-summarization",
    "text-generation",
    "image-classification",
    "image-segmentation",
    "object-detection",
    "text-to-image",
    "zero-shot-image-classification",
    "table-question-answering",
    "zero-shot-classification",
    "visual-question-answering",
    "image-to-text"
]

def get_trending_models_and_datasets():
    """Fetch top 200 trending models from Hugging Face Hub."""
    hf_api = HfApi()
    
    logger.info("Fetching top 200 trending models...")
    models = hf_api.list_models(
        sort="trendingScore",
        direction=-1,
        limit=200,
        full=True
    )
    return models

def send_slack_message(message: str):
    """Send a message to Slack using webhook URL"""
    if not SLACK_WEBHOOK_URL:
        logger.warning(f"No Slack webhook URL configured. Message: {message}")
        return
    
    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        logger.info(f"Slack message sent successfully: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Slack message: {e}")

def is_safetensors_bot_pr(model_id):
    """Find the latest open PR by SFconvertbot for the given model"""
    try:
        discussions = list(get_repo_discussions(
            repo_id=model_id,
            author="SFconvertbot",
            discussion_type="pull_request",
            discussion_status="open"
        ))
        
        # Find the most recent PR
        return discussions
    except Exception as e:
        logger.error(f"Error fetching PR for {model_id}: {str(e)}")
        return None

def is_model_in_catalog(model_id):
    """Check if the model is in the Azure Model Catalog"""
    response = requests.get(requests.get(
        url="https://generate-azureml-urls.azurewebsites.net/api/generate", 
        params={"modelId": model_id})
    )
    return response.status_code == 200

def is_security_scanned(model_id):
    """Check security status for a given model."""
    url = f"https://huggingface.co/api/models/{model_id}?securityStatus=1&expand[]=sha"
    
    try:
        logger.info(f"Checking security status for {model_id}")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Check if securityRepoStatus exists
            if 'securityRepoStatus' in data:
                security_status = data['securityRepoStatus']
                if security_status.get('scansDone', False) is False:
                    return None
                is_secure = (
                    security_status.get('scansDone', False) is True and 
                    len(security_status.get('filesWithIssues', [])) == 0
                )
                return is_secure
            
            # Check if nested somewhere else
            for key, value in data.items():
                if isinstance(value, dict) and 'securityRepoStatus' in value:
                    security_status = value['securityRepoStatus']
                    is_secure = (
                        security_status.get('scansDone', False) is True and 
                        len(security_status.get('filesWithIssues', [])) == 0
                    )
                    return is_secure
                    
        return None  # Security status not found or request failed
    except Exception as e:
        logger.error(f"Error checking {model_id}: {str(e)}")
        return None

def prepare_model_data(models):
    """Prepare model data for the dataset."""
    model_data = []
    for model in models:
        author = model.modelId.split('/')[0] if '/' in model.modelId else ""
        license = [tag for tag in model.tags if tag.startswith("license:")]
        
        model_data.append({
            ## raw data
            "id": model.modelId,
            "type": "model",
            "author": author,
            "downloads": model.downloads if hasattr(model, 'downloads') else 0,
            "likes": model.likes if hasattr(model, 'likes') else 0,
            "tags": model.tags if hasattr(model, 'tags') else [],
            "last_modified": model.lastModified,
            "created_at": model.createdAt if hasattr(model, 'createdAt') else None,
            "sha": model.sha,
            'license': license,
            'library_name': model.library_name,
            'gated': model.gated,
            ## logic to check supported prerequisites
            'is_in_catalog' : is_model_in_catalog(model.modelId),
            'is_custom_code': 'custom_code' in model.tags,
            'is_excluded_org' : author in EXCLUDED_ORGS, 
            'is_supported_license' : bool(license) and any(l in ALLOWED_LICENSES for l in license),
            'is_supported_library' : model.library_name in ['diffusers', 'transformers'],
            'is_safetensors': 'safetensors' in model.tags or is_safetensors_bot_pr(model.modelId),
            'is_supported_task' : model.pipeline_tag in SUPPORTED_TASKS,
            'is_securely_scanned' : is_security_scanned(model.modelId),
            "collected_at": COLLECTION_DATE
        })

    for model in model_data:
        if model['is_in_catalog']:
            model['model_status'] = 'added'
        elif all([
            not model['is_custom_code'],
            not model['is_excluded_org'],
            model['is_supported_license'],
            model['is_supported_library'],
            model['is_safetensors'],
            model['is_supported_task'],
            model['is_securely_scanned']
        ]):
            model['model_status'] = 'to add'
        else:
            model['model_status'] = 'blocked'

    return pd.DataFrame(model_data)


def update_dataset(models_df, dataset_repo):
    """Update or create the dataset with new data."""
    from datasets import load_dataset, DatasetDict
    
    try:
        # Try to load existing dataset
        existing_ds = load_dataset(dataset_repo)
        
        # Append new data to existing splits
        existing_models = Dataset.from_pandas(existing_ds["models"].to_pandas())
        
        new_models = concatenate_datasets([existing_models, Dataset.from_pandas(models_df)])
    except Exception as e:
        logger.info(f"Dataset doesn't exist or couldn't be loaded, creating new one: {e}")
        new_models = Dataset.from_pandas(models_df)
    
    # Create dataset with splits
    dataset = DatasetDict({
        "models": new_models
    })
    
    # Push to Hub
    dataset.push_to_hub(dataset_repo)
    logger.info(f"Successfully updated dataset at {dataset_repo}")


def main():
    # Configuration
    if len(sys.argv) < 2:
        logger.error("Usage: python trending_analysis.py <DATASET_REPO>")
        return

    DATASET_REPO = sys.argv[1]

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
        logger.error("Exiting due to authentication issue. Please run 'huggingface-cli login'.")
        return # Exit the main function if not logged in.

    # Get data
    models = get_trending_models_and_datasets()
    
    # Prepare DataFrames
    models_df = prepare_model_data(models)
    
    # Update dataset
    update_dataset(models_df, DATASET_REPO)

    # Send Slack message
    # message = (
    #     "📈 Trending Models Analysis 📈\n\n"
    #     f"To add: {models_df[models_df['model_status'] == 'to add'].shape[0]}\n"
    #     f"Blocked: {models_df[models_df['model_status'] == 'blocked'].shape[0]}\n"
    #     f"Added: {models_df[models_df['model_status'] == 'added'].shape[0]}\n"
    #     f"View details: https://hf.co/spaces/hf-azure-internal/trending-models-analysis"
    # )
    # send_slack_message(message)


if __name__ == "__main__":
    main()