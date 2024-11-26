from datasets import load_dataset
from typing import List, Dict
from config import Config


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_reference_datasets(self) -> List[Dict]:
        qa_pairs = []
        i = 0
        for dataset_name, config in self.config.reference_datasets:
            dataset = load_dataset(dataset_name, config, split="train")
            for item in dataset:
                messages = item["messages"]
                # Extract Q&A from messages
                question = next(m["content"] for m in messages if m["role"] == "user")
                answer = next(
                    m["content"] for m in messages if m["role"] == "assistant"
                )
                qa_pairs.append({"id": str(i), "question": question, "answer": answer})
                i += 1

        return qa_pairs

    def load_experiment_dataset(self):
        dataset_name, config = self.config.experiment_dataset
        return load_dataset(dataset_name, config, split="train")
