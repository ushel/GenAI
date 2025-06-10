

# from rag import get_answer

# from scorer import score_answer
import sys
import os
# import unittest

# # Get the directory of the current script (test_main.py)
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Go up one level to RAG_module directory
# rag_module_dir = os.path.join(current_dir, '..')

# # Add RAG_module directory to sys.path
# sys.path.insert(0, rag_module_dir)

from rag import get_answer
from scorer import score_answer
def test_rag_pipeline():
    question = "What is MetaGPT?"
    answer = get_answer(question)
    assert isinstance(answer, str)
