from huggingface_hub import hf_hub_download, upload_file
import os

def download_from_hf(repo_id: str, filename: str, target_path: str, repo_type: str = "model"):
    """
    Downloads a file from a Hugging Face repository and saves it to the exact target_path.

    Args:
        repo_id (str): The ID of the repository (e.g., 'bert-base-uncased').
        filename (str): The name of the file in the repo (e.g., 'config.json').
        target_path (str): The exact local path where the file should be saved.
        repo_type (str): The type of the repository ('model', 'dataset', or 'space'). Defaults to 'model'.

    Returns:
        str: The local path to the downloaded file.
    """
    save_dir = os.path.dirname(target_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Download to local_dir (which will mirror repo structure)
    downloaded_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_dir,
        repo_type=repo_type
    )
    
    # If the file wasn't downloaded exactly to target_path (due to mirroring), move it
    if os.path.abspath(downloaded_file_path) != os.path.abspath(target_path):
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(downloaded_file_path, target_path)
        print(f"Moved {downloaded_file_path} to {target_path}")
        
        # Clean up empty subdirectories created by hf_hub_download
        current_dir = os.path.dirname(downloaded_file_path)
        while current_dir and current_dir.startswith(os.path.abspath(save_dir)) and current_dir != os.path.abspath(save_dir):
            try:
                os.rmdir(current_dir)
                current_dir = os.path.dirname(current_dir)
            except OSError:
                break  # Directory not empty or other error
    
    print(f"Downloaded {filename} from {repo_id} to {target_path}")
    return target_path
