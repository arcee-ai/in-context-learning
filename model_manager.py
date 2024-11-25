from transformers import AutoTokenizer, AutoModel, pipeline
import torch

from vllm import LLM, SamplingParams, RequestOutput

# import torch
from typing import List, Dict, Tuple
from config import Config


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.model = LLM(
            model=config.llm_model, gpu_memory_utilization=0.7, max_model_len=16000
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )

        self.ranking_model = AutoModel.from_pretrained(
            self.config.reward_model,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.ranking_tokenizer = AutoTokenizer.from_pretrained(
            self.config.reward_model, trust_remote_code=True
        )

    def generate_response(self, prompt: str, context_qa: Dict = None) -> str:
        if context_qa:
            messages = [
                {"role": "user", "content": context_qa["question"]},
                {"role": "assistant", "content": context_qa["answer"]},
                {"role": "user", "content": prompt},
            ]
            # Convert messages to the format expected by your specific model

        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        responses: List[RequestOutput] = self.model.generate(
            text,
            self.sampling_params,
        )

        # Generate response using the model
        return responses[0].outputs[0].text

    def compare_responses(
        self, response1: List[dict], response2: List[dict]
    ) -> List[float]:
        scores = self.ranking_model.get_scores(
            self.ranking_tokenizer, [response1, response2]
        )
        return scores
