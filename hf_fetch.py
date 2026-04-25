from huggingface_hub import hf_hub_download, upload_file
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

def upload_to_hf(repo_id: str, file_path: str, path_in_repo: str, repo_type: str = "model", token: str = None):
    """
    Uploads a file to a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository (e.g., 'username/repo-name').
        file_path (str): The local path to the file to upload.
        path_in_repo (str): The path where the file should be saved in the repository.
        repo_type (str): The type of the repository ('model', 'dataset', or 'space'). Defaults to 'model'.
        token (str, optional): Hugging Face API token. If not provided, it will look for a logged-in session.

    Returns:
        str: The URL of the uploaded file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    uploaded_url = upload_file(
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token
    )

    print(f"Uploaded {file_path} to {repo_id} at {path_in_repo}")
    return uploaded_url

def download_all_hf_files():
    ## OracleNet Files
    download_from_hf("VP1606/OracleNet", "best_model.pt", "./final_model_versions/ff2f02d4")
    download_from_hf("VP1606/OracleNet", "meta.json", "./final_model_versions/ff2f02d4")
    
    ## ELMo Files
    download_from_hf("VP1606/OracleNet", "elmo/weights.hdf5", "./notebook_data/elmo_model/weights.hdf5")
    download_from_hf("VP1606/OracleNet", "elmo/options.json", "./notebook_data/elmo_model/options.json")
    
    ## Meta PT
    download_from_hf("VP1606/OracleNet", "notebook_data/meta.pt", "embeddings/meta.pt")
    
    ## ELMO Weights
    download_from_hf("VP1606/OracleNet", "notebook_data/elmo_model/options.json", "elmo/options.json")
    download_from_hf("VP1606/OracleNet", "notebook_data/elmo_model/weights.hdf5", "elmo/weights.hdf5")

if __name__ == "__main__":
    # download_all_hf_files()
    
    ## Upload Meta PT
    # upload_to_hf("VP1606/OracleNet", "notebook_data/meta.pt", "embeddings/meta.pt")
    # upload_to_hf("VP1606/OracleNet", "notebook_data/elmo_model/options.json", "elmo/options.json")
    # upload_to_hf("VP1606/OracleNet", "notebook_data/elmo_model/weights.hdf5", "elmo/weights.hdf5")
    
    pass