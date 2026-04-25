from huggingface_hub import hf_hub_download
import os

def download_from_hf(repo_id: str, filename: str, save_path: str, repo_type: str = "model"):
    """
    Downloads a file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository (e.g., 'bert-base-uncased').
        filename (str): The name of the file to download (e.g., 'config.json').
        save_path (str): The local directory where the file should be saved.
        repo_type (str): The type of the repository ('model', 'dataset', or 'space'). Defaults to 'model'.

    Returns:
        str: The local path to the downloaded file.
    """
    os.makedirs(save_path, exist_ok=True)
    
    downloaded_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=save_path,
        repo_type=repo_type
    )
    
    print(f"Downloaded {filename} from {repo_id} to {downloaded_file_path}")
    return downloaded_file_path

if __name__ == "__main__":
    # Example usage:
    # download_from_hf("bert-base-uncased", "config.json", "./models/bert")
    download_from_hf("VP1606/OracleNet", "best_model.pt", "./final_model_versions/ff2f02d4")
    download_from_hf("VP1606/OracleNet", "meta.json", "./final_model_versions/ff2f02d4")