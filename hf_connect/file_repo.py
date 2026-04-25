from hf_connect.manager import ModelFileManager

"""
CLI Reference Key Table

ORACLE_NET_FILES: oracle_net
ELMO_FILES: elmo
MODERN_BERTA_FILES: modern_berta
TASK_SOURCE_FILES: task_source
"""

ORACLE_NET_FILES = [
    ModelFileManager("best_model.pt", "./final_model_versions/ff2f02d4/best_model.pt"),
    ModelFileManager("meta.json", "./final_model_versions/ff2f02d4/meta.json")
]

ELMO_FILES = [
    ModelFileManager("elmo/weights.hdf5", "./notebook_data/elmo_model/weights.hdf5"),
    ModelFileManager("elmo/options.json", "./notebook_data/elmo_model/options.json"),
    ModelFileManager("embeddings/meta.pt", "./notebook_data/meta.pt")
]

MODERN_BERTA_FILES = [
    ModelFileManager("oracle_tf/modern_berta/model.safetensors", "./final_model_versions/modernBerta/model.safetensors"),
    
    ModelFileManager("oracle_tf/modern_berta/config.json", "./final_model_versions/modernBerta/config.json"),
    ModelFileManager("oracle_tf/modern_berta/tokenizer_config.json", "./final_model_versions/modernBerta/tokenizer_config.json"),
    ModelFileManager("oracle_tf/modern_berta/tokenizer.json", "./final_model_versions/modernBerta/tokenizer.json")
]

TASK_SOURCE_FILES = [
    ModelFileManager("oracle_tf/task_source/model.safetensors", "./final_model_versions/taskSource/model.safetensors"),
    
    ModelFileManager("oracle_tf/modern_berta/config.json", "./final_model_versions/modernBerta/config.json"),
    ModelFileManager("oracle_tf/modern_berta/tokenizer_config.json", "./final_model_versions/modernBerta/tokenizer_config.json"),
    ModelFileManager("oracle_tf/modern_berta/tokenizer.json", "./final_model_versions/modernBerta/tokenizer.json")
]