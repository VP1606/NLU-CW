from hf_connect.push import upload_to_hf
from hf_connect.pull import download_from_hf
import os

class ModelFileManager:
    def __init__(
        self,
        remote_path: str,
        local_path: str,
        repo_id: str = "VP1606/OracleNet"
    ):
        self.remote_path = remote_path
        self.local_path = local_path
        self.repo_id = repo_id
        
    def pull(self):
        return download_from_hf(
            repo_id=self.repo_id,
            filename=self.remote_path,
            save_path=os.path.dirname(self.local_path)
        )
    
    def push(self):
        return upload_to_hf(
            repo_id=self.repo_id,
            file_path=self.local_path,
            path_in_repo=self.remote_path
        )
