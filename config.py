from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Config:
    # Dataset configurations
    reference_datasets = [
        ("mlabonne/orca-agentinstruct-1M-v1-cleaned", "default"),
        ("HuggingFaceTB/smoltalk", "all"),
    ]
    experiment_dataset = "lmsys/lmsys-chat-1m"

    # Database configurations
    vector_db_path = "vector_store"
    sql_db_path = "sqlite:///answers.db"
    results_db_path = "sqlite:///results.db"

    # Model configurations
    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "/media/teamgroup/models/Qwen2.5-3B-Instruct"
    reward_model = "/media/teamgroup/models/internlm2-1_8b-reward"
    pairwise_ranker = "mlabonne/pairRM"

    # Experiment parameters
    batch_size = 256
    num_neighbors = 1
