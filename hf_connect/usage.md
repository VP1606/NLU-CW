# Hugging Face Connect CLI Usage

This tool allows you to easily push and pull model files and assets between your local environment and Hugging Face Hub.

## Commands

The tool is designed to be run as a module from the root directory:

```bash
python3 -m hf_connect.main [OPTIONS] [REPO_KEY ...]
```

### Options

- `--push`: Upload local files to the Hugging Face repository.
- `--pull`: Download files from the Hugging Face repository to your local machine.
- `--all_files`: Perform the operation for all configured repository groups.
- `-h, --help`: Show the help message.

### Repository Reference Keys

Instead of manually specifying paths, you can use the following shorthand keys:

| Reference Key | Description |
| :--- | :--- |
| `oracle_net` | Oracle Net model and metadata |
| `elmo` | ELMo weights, options, and metadata |
| `modern_berta` | ModernBERT-Large safe tensors |
| `task_source` | TaskSource safe tensors |

## Examples

### Push specific files
To upload the ELMo model files:
```bash
python3 -m hf_connect.main --push elmo
```

### Push multiple repositories
To upload both ELMo and Oracle Net files:
```bash
python3 -m hf_connect.main --push elmo oracle_net
```

### Pull specific files
To download the Oracle Net files:
```bash
python3 -m hf_connect.main --pull oracle_net
```

### Sync everything
To download all files defined in the repository:
```bash
python3 -m hf_connect.main --pull --all_files
```

## Configuration

File mappings and reference keys are managed in `hf_connect/file_repo.py`. Each group is a list of `ModelFileManager` objects defining the remote path, local path, and target Hugging Face repository ID.
