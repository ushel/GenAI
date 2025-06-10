from rag import get_answer
from scorer import score_answer

if __name__ == "__main__":
    question = "what are keys values and query?"
    answer = get_answer(question)
    score = score_answer(question, answer)
    print(f"Answer: {answer}")
    print(f"Score: {score}")
