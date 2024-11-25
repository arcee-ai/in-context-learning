import pytest
import numpy as np
from config import Config
from database import DatabaseManager
from parallel_db_manager import ParallelDatabaseManager
import shutil
import os
import torch


@pytest.fixture
def test_data():
    return [
        {"id": "1", "question": "What is Python?", "answer": "A programming language"},
        {
            "id": "2",
            "question": "What is TensorFlow?",
            "answer": "A machine learning framework",
        },
        {
            "id": "3",
            "question": "What is PyTorch?",
            "answer": "A deep learning framework",
        },
        {
            "id": "4",
            "question": "What is CUDA?",
            "answer": "A parallel computing platform",
        },
        {"id": "5", "question": "What is SQL?", "answer": "A database query language"},
    ]


@pytest.fixture
def config():
    test_config = Config()
    test_config.vector_db_path = "test_vector_store"
    return test_config


def cleanup_test_db(path):
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_database_implementations(config, test_data):
    # Clean up any existing test databases
    cleanup_test_db(config.vector_db_path)

    try:
        # Initialize both implementations
        old_db = DatabaseManager(config)
        new_db = ParallelDatabaseManager(config)

        # Store the same data in both
        old_db.store_qa_pairs(test_data)
        new_db.store_qa_pairs(test_data)

        # Test queries with different types of questions
        test_queries = [
            "Tell me about Python programming",
            "Explain machine learning frameworks",
            "What's CUDA used for?",
            "Database query languages",
            "Deep learning tools",
        ]

        for query in test_queries:
            old_q, old_a = old_db.find_similar_question(query)
            new_q, new_a = new_db.find_similar_question(query)

            # Compare results
            assert old_q is not None, f"Old DB returned None for query: {query}"
            assert new_q is not None, f"New DB returned None for query: {query}"
            assert old_a is not None, f"Old DB returned None answer for query: {query}"
            assert new_a is not None, f"New DB returned None answer for query: {query}"

            # We can't guarantee exact matches due to parallel processing potentially
            # affecting the order of embeddings, but we can check that both implementations
            # return valid results
            assert any(
                q["question"] == new_q for q in test_data
            ), f"New DB returned invalid question for query: {query}"
            assert any(
                q["answer"] == new_a for q in test_data
            ), f"New DB returned invalid answer for query: {query}"
            assert old_q == new_q, f"Different questions found {old_q} {new_q}"

    finally:
        # Clean up
        cleanup_test_db(config.vector_db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
