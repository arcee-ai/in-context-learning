from typing import Dict, Generator, Iterable, List, Tuple
import tqdm
from config import Config
from database import DatabaseManager
from model_manager import ModelManager
from data_loader import DataLoader
from itertools import islice
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio


@dataclass
class ExperimentItem:
    question: str
    similar_question: str
    similar_answer: str


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

    async def process_batch(self, batch: List[Dict]) -> None:
        # Prepare experiment items with similar questions
        items = []
        for item in batch:
            question = item["conversation"][0]["content"]
            similar_q, similar_a = self.db_manager.find_similar_question(question)
            items.append(
                ExperimentItem(
                    question=question,
                    similar_question=similar_q,
                    similar_answer=similar_a,
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

            # Prepare messages for scoring
            context_message = [
                {"role": "user", "content": item.question},
                {"role": "assistant", "content": with_context_answer},
            ]
            no_context_message = [
                {"role": "user", "content": item.question},
                {"role": "assistant", "content": without_context_answer},
            ]

            # Compare responses
            scores = self.model_manager.compare_responses(
                context_message, no_context_message
            )

            # Store results
            self.db_manager.store_result(
                {
                    "question": item.question,
                    "context_qa": {
                        "question": item.similar_question,
                        "answer": item.similar_answer,
                    },
                    "with_context_answer": with_context_answer,
                    "without_context_answer": without_context_answer,
                    "with_context_score": scores[0],
                    "without_context_score": scores[1],
                    "with_context_better": scores[0] > scores[1],
                }
            )

    async def run(self):
        # Load experiment dataset
        experiment_data = self.data_loader.load_experiment_dataset()
        # Process in batches
        tasks = []
        for batch in tqdm.tqdm(
            self.batch_items(experiment_data, self.batch_size),
            desc="Processing batches",
            total=(len(experiment_data) + self.batch_size - 1) // self.batch_size,
        ):
            task = asyncio.create_task(self.process_batch(batch))
            tasks.append(task)
            if len(tasks) >= 3:  # Limit concurrent batches to avoid OOM
                await asyncio.gather(*tasks)
                tasks = []

        # Process any remaining tasks
        if tasks:
            await asyncio.gather(*tasks)