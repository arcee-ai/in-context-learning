from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import llm_blender
from vllm import LLM, SamplingParams, RequestOutput

# import torch
from typing import List, Dict, Tuple
from config import Config


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.sampling_params = None

        self.ranking_model = None
        self.ranking_tokenizer = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self.model = LLM(
            model=self.config.llm_model, gpu_memory_utilization=0.7, max_model_len=25000
        )
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=4096,
        )

    def unload_model(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def load_ranker(self):
        if self.config.internlm:
            self.ranking_model = AutoModel.from_pretrained(
                self.config.reward_model,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.ranking_tokenizer = AutoTokenizer.from_pretrained(
                self.config.reward_model, trust_remote_code=True
            )
        else:
            self.ranking_model = llm_blender.Blender()
            self.ranking_model.loadranker("llm-blender/PairRM")

    def unload_ranker(self):
        del self.ranking_model
        del self.ranking_tokenizer
        torch.cuda.empty_cache()

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
        self, question: str, with_context_answer: str, without_context_answer: str
    ) -> List[float]:

        context_message = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": with_context_answer},
        ]
        no_context_message = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": without_context_answer},
        ]
        if self.config.internlm:
            scores = self.ranking_model.get_scores(
                self.ranking_tokenizer, [context_message, no_context_message]
            )
        else:
            scores = self.ranking_model.rank(
                [question],
                [[with_context_answer, without_context_answer]],
                return_scores=True,
                disable_tqdm=True,
            )[0]

        return scores
