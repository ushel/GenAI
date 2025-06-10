
# def score_answer(question: str, answer: str) -> float:
#     question_keywords = [word.lower() for word in question.split() if len(word) > 3]
#     matched = sum(1 for word in question_keywords if word in answer.lower())
#     return matched / len(question_keywords) if question_keywords else 0.0

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def score_answer(question: str, answer: str) -> float:
   
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)


    similarity_score = util.cos_sim(question_embedding, answer_embedding).item()

   
    return similarity_score