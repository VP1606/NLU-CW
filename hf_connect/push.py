from huggingface_hub import hf_hub_download, upload_file
import os

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
