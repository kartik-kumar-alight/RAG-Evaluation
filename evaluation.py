from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score

# Example data
retrieved_docs = [
    ["doc1", "doc2", "doc3"], 
    ["doc2", "doc3", "doc4"]
]  # List of retrieved documents for each query
relevant_docs = [
    ["doc1", "doc3"], 
    ["doc3", "doc4"]
]  # List of relevant documents for each query
generated_texts = [
    "This is the generated answer for query 1.",
    "This is the generated answer for query 2."
]  # List of generated texts for each query
reference_texts = [
    "This is the reference answer for query 1.",
    "This is the reference answer for query 2."
]  # List of reference texts for each query

# 1. Retrieval Metrics
def precision_at_k(retrieved, relevant, k):
    return np.mean([len(set(retrieved[i][:k]) & set(relevant[i])) / k for i in range(len(retrieved))])

def recall_at_k(retrieved, relevant, k):
    return np.mean([len(set(retrieved[i][:k]) & set(relevant[i])) / len(set(relevant[i])) for i in range(len(retrieved))])

def mean_reciprocal_rank(retrieved, relevant):
    rr = []
    for i in range(len(retrieved)):
        for rank, doc in enumerate(retrieved[i]):
            if doc in relevant[i]:
                rr.append(1 / (rank + 1))
                break
        else:
            rr.append(0)
    return np.mean(rr)

# Calculate retrieval metrics
k = 3
precision_k = precision_at_k(retrieved_docs, relevant_docs, k)
recall_k = recall_at_k(retrieved_docs, relevant_docs, k)
mrr = mean_reciprocal_rank(retrieved_docs, relevant_docs)

print(f'Precision@{k}: {precision_k}')
print(f'Recall@{k}: {recall_k}')
print(f'MRR: {mrr}')

# 2. Generation Metrics
def bleu_score(generated, references):
    return np.mean([sentence_bleu([ref.split()], gen.split()) for gen, ref in zip(generated, references)])

def rouge_scores(generated, references):
    rouge = Rouge()
    scores = rouge.get_scores(generated, references, avg=True)
    return scores

def bert_scores(generated, references):
    P, R, F1 = bert_score(generated, references, lang='en', rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

# Calculate generation metrics
bleu = bleu_score(generated_texts, reference_texts)
rouge = rouge_scores(generated_texts, reference_texts)
bert_P, bert_R, bert_F1 = bert_scores(generated_texts, reference_texts)

print(f'BLEU Score: {bleu}')
print(f'ROUGE Scores: {rouge}')
print(f'BERT Precision: {bert_P}, Recall: {bert_R}, F1: {bert_F1}')

# 3. Joint Evaluation Metrics
def exact_match(generated, references):
    return np.mean([gen.strip() == ref.strip() for gen, ref in zip(generated, references)])

def f1_score_joint(generated, references):
    generated_tokens = [set(gen.split()) for gen in generated]
    reference_tokens = [set(ref.split()) for ref in references]
    precision = np.mean([len(gen & ref) / len(gen) if gen else 0 for gen, ref in zip(generated_tokens, reference_tokens)])
    recall = np.mean([len(gen & ref) / len(ref) if ref else 0 for gen, ref in zip(generated_tokens, reference_tokens)])
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Calculate joint metrics
em = exact_match(generated_texts, reference_texts)
f1 = f1_score_joint(generated_texts, reference_texts)

print(f'Exact Match: {em}')
print(f'F1 Score: {f1}')
