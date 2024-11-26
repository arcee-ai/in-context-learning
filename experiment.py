from typing import Dict, Generator, Iterable, List, Tuple
import tqdm
from config import Config
from database import DatabaseManager, Result
from model_manager import ModelManager
from data_loader import DataLoader
from itertools import islice
from dataclasses import dataclass


@dataclass
class ExperimentItem:
    question: str
    similar_question: str
    similar_answer: str
    score: float


class OptimizedExperiment:
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.data_loader = DataLoader(config)
        self.model_manager = ModelManager(config)
        self.batch_size = config.batch_size

    def batch_items(self, iterable: Iterable, batch_size: int) -> Generator:
        iterator = iter(iterable)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    def setup(self):
        # Load and store reference datasets
        qa_pairs = self.data_loader.load_reference_datasets()

        for batch in tqdm.tqdm(
            self.batch_items(qa_pairs, self.batch_size),
            desc="Storing QA pairs",
            total=(len(qa_pairs) + self.batch_size - 1) // self.batch_size,
        ):
            self.db_manager.store_qa_pairs(batch)

    def prepare_prompts(
        self, items: List[ExperimentItem]
    ) -> Tuple[List[str], List[str]]:
        """Prepare batched prompts for both with-context and without-context scenarios"""
        with_context_prompts = []
        without_context_prompts = []

        for item in items:
            # Prepare with-context prompt
            with_context_messages = [
                {"role": "user", "content": item.similar_question},
                {"role": "assistant", "content": item.similar_answer},
                {"role": "user", "content": item.question},
            ]
            with_context_prompt = self.model_manager.tokenizer.apply_chat_template(
                with_context_messages, tokenize=False, add_generation_prompt=True
            )
            with_context_prompts.append(with_context_prompt)

            # Prepare without-context prompt
            without_context_messages = [
                {"role": "user", "content": item.question},
            ]
            without_context_prompt = self.model_manager.tokenizer.apply_chat_template(
                without_context_messages, tokenize=False, add_generation_prompt=True
            )
            without_context_prompts.append(without_context_prompt)

        return with_context_prompts, without_context_prompts

    async def process_generation(self, batch: List[Dict]) -> None:
        items = []
        for item in batch:
            question = item["messages"][0]["content"]
            similar_q, similar_a, score = self.db_manager.find_similar_question(
                question
            )
            items.append(
                ExperimentItem(
                    question=question,
                    similar_question=similar_q,
                    similar_answer=similar_a,
                    score=score,
                )
            )

        # Prepare batched prompts
        with_context_prompts, without_context_prompts = self.prepare_prompts(items)

        # Generate responses in batches
        with_context_responses = self.model_manager.model.generate(
            with_context_prompts,
            self.model_manager.sampling_params,
        )
        without_context_responses = self.model_manager.model.generate(
            without_context_prompts,
            self.model_manager.sampling_params,
        )

        # Process results and store them
        for item, with_context, without_context in zip(
            items, with_context_responses, without_context_responses
        ):
            with_context_answer = with_context.outputs[0].text
            without_context_answer = without_context.outputs[0].text

            # Store results
            self.db_manager.store_result(
                {
                    "question": item.question,
                    "context_score": item.score,
                    "context_qa": {
                        "question": item.similar_question,
                        "answer": item.similar_answer,
                    },
                    "with_context_answer": with_context_answer,
                    "without_context_answer": without_context_answer,
                    "with_context_score": None,
                    "without_context_score": None,
                    "with_context_better": None,
                    "processed": False,
                }
            )

    async def score_and_update(self, item: Result) -> None:
        question = item.question
        with_context_answer = item.with_context_answer
        without_context_answer = item.without_context_answer

        if len(with_context_answer) < 5000 or len(without_context_answer) < 5000:
            return

        # Compare responses
        scores = self.model_manager.compare_responses(
            question, with_context_answer, without_context_answer
        )

        # Update the existing record with scores
        self.db_manager.update_result(item.id, scores)

    async def run(self):
        # Load experiment dataset
        experiment_data = self.data_loader.load_experiment_dataset()

        self.model_manager.load_model()
        # Process in batches
        tasks = []
        for batch in tqdm.tqdm(
            self.batch_items(experiment_data, self.batch_size),
            desc="Generating",
            total=(len(experiment_data) + self.batch_size - 1) // self.batch_size,
        ):
            await self.process_generation(batch)

        self.model_manager.unload_model()

        self.model_manager.load_ranker()

        items = self.db_manager.retrieve_results()

        for item in tqdm.tqdm(items, desc="Ranking", total=len(items)):
            await self.score_and_update(item)

        self.model_manager.unload_ranker()
