import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge_score.rouge_scorer import RougeScorer

nltk.download('wordnet')


def get_scores(references: [[str]], candidate: [str]):
    return {
        "bleu": _get_bleu_score(references, candidate),
        "meteor": meteor_score(references, candidate),
        "nist": _get_nist_score(references, candidate),
        "rouge": _get_rouge_score(references, candidate)
    }


def _get_bleu_score(references: [[str]], candidate: [str]):
    weights = [
        (0.25, 0.25, 0.25, 0.25),  # BLEU
        (1, 0, 0, 0),  # BLEU-1
        (0, 1, 0, 0),  # BLEU-2
        (0, 0, 1, 0),  # BLEU-3
        (0, 0, 0, 1),  # BLEU-4
    ]
    labels = ['BLEU', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', ]

    def _eval(w):
        try:
            return sentence_bleu(references, candidate, weights=w)
        except ZeroDivisionError:
            return None

    return {l: _eval(w) for l, w in zip(labels, weights)}


def _get_nist_score(references, candidate):
    try:
        return sentence_nist(references, candidate)
    except ZeroDivisionError:
        return 0


def _get_rouge_score(references, candidate):
    candidate_str = ' '.join(candidate)
    r_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                           use_stemmer=True)
    return [r_scorer.score(' '.join(i), candidate_str) for i in references]
