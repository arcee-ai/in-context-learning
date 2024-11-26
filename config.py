from dataclasses import dataclass

@dataclass
class Config:
    # Dataset configurations
    reference_datasets = [
        ("mlabonne/orca-agentinstruct-1M-v1-cleaned", "default"),
    ]
    experiment_dataset = ("HuggingFaceTB/smoltalk", "default")

    # Database configurations
    vector_db_path = "vector_store"
    results_db_path = "sqlite:///results.db"

    # Model configurations
    embedding_model = "BAAI/bge-small-en-v1.5"
    llm_model = "Qwen/Qwen2.5-7B-Instruct"
    reward_model = "internlm/internlm2-7b-reward"

    # Experiment parameters
    batch_size = 256
    num_neighbors = 1

    internlm = False
