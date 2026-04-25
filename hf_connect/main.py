import argparse
from hf_connect import file_repo

def main():
    parser = argparse.ArgumentParser(description="Hugging Face File Manager CLI")
    
    # Push/Pull selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--push", action="store_true", help="Push files to Hugging Face")
    group.add_argument("--pull", action="store_true", help="Pull files from Hugging Face")
    
    # Repo selection
    # Reference keys from file_repo.py comments:
    # oracle_net, elmo, modern_berta, task_source
    parser.add_argument("repos", nargs="*", choices=["oracle_net", "elmo", "modern_berta", "task_source"], 
                        help="Which repo(s) to use (reference key)")
    parser.add_argument("--all_files", action="store_true", help="Process all repos")

    args = parser.parse_args()

    # Mapping based on CLI Reference Key Table in file_repo.py
    repo_map = {
        "oracle_net": file_repo.ORACLE_NET_FILES,
        "elmo": file_repo.ELMO_FILES,
        "modern_berta": file_repo.MODERN_BERTA_FILES,
        "task_source": file_repo.TASK_SOURCE_FILES,
    }

    files_to_process = []
    if args.all_files:
        for repo_files in repo_map.values():
            files_to_process.extend(repo_files)
    elif args.repos:
        for repo_key in args.repos:
            files_to_process.extend(repo_map[repo_key])
    else:
        parser.print_help()
        print("\nError: You must specify at least one repo reference key or use --all_files")
        return

    for file_manager in files_to_process:
        if args.push:
            file_manager.push()
        elif args.pull:
            file_manager.pull()

if __name__ == "__main__":
    main()
