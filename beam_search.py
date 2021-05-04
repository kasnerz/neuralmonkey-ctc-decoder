from typing import List
import numpy as np
from scipy.misc import logsumexp
import copy
import pprint as pp

from n_gram_model import NGramModel
from neuralmonkey.vocabulary import Vocabulary

from hypothesis import Hypothesis, ExpandFunction


def empty_hypothesis() -> Hypothesis:
    return Hypothesis([], 0.0, 0.0, None)

def expand_null(prev: Hypothesis, null_score: float) -> Hypothesis:
    """Expand with null CTC token.

    CTC score gets updated, but the LM score and state remain the same.
    """
    hyp = copy.deepcopy(prev)
    hyp.expand_by_null(null_score)
    return hyp


def log_softmax(logits_table):
    return logits_table - logsumexp(logits_table, axis=1, keepdims=True)


def score_hypothesis(hyp : Hypothesis, weights: dict, states_cnt: int):
    if (len(hyp.tokens) + hyp.null_total) == 0 :
        return -1e9

    score = 0

    for key in weights.keys():
        score += weights[key] * compute_feature(key, hyp, states_cnt)

    score += compute_feature("ctc_score", hyp, states_cnt)

    return score

def compute_feature(key: str, hyp: Hypothesis, states_cnt: int):
    num_of_tokens = len(hyp.tokens)
    k = 3

    if key == 'ctc_score':
        return (hyp.ctc_score / (num_of_tokens + hyp.null_total) 
                    if (num_of_tokens + hyp.null_total) else -1)

    elif key == 'lm_score':
        return hyp.lm_score / num_of_tokens if num_of_tokens else -2

    elif key == 'null_token_ratio':
        return (max(abs((hyp.null_total/num_of_tokens) - (2/1))-2,0) 
                    if num_of_tokens else 0)

    elif key == 'null_trailing':
        return max(hyp.null_trailing-(states_cnt/k),0)

def decode_beam(
        logits_table: np.ndarray,
        beam_width: int,
        vocabulary: Vocabulary,
        lm: NGramModel,
        weights: dict) -> List[str]:
    assert beam_width >= 1
    
    log_prob_table = log_softmax(logits_table)

    hypotheses = [empty_hypothesis()]
    states_cnt = len(log_prob_table)

    for time, log_probs in enumerate(log_prob_table):
        null_log_prob = log_probs[-1]
        token_log_probs = log_probs[:-1]
        new_hypotheses = []

        str_to_hyp = {}

        for hyp in hypotheses:
            expanded = expand_null(hyp, null_log_prob)
            str_to_hyp[" ".join(expanded.tokens)] = (
                expanded, len(new_hypotheses))
            new_hypotheses.append(expanded)

        best_tokens = np.argpartition(
            -token_log_probs, 2 * beam_width)[:2 * beam_width]
        best_scores = token_log_probs[best_tokens]

        for hyp in hypotheses:
            for index, score in zip(best_tokens, best_scores):
                token = vocabulary.index_to_word[index]
                expanded = lm.expand_token(hyp, token, score)
                score = score_hypothesis(expanded, weights, states_cnt)
                hyp_str = " ".join(expanded.tokens)
                if hyp_str in str_to_hyp:
                    orig_hyp, index = str_to_hyp[hyp_str]
                    expanded.recombine_with(orig_hyp)
                    new_hypotheses[index] = expanded
                    str_to_hyp[hyp_str] = (expanded, index)
                else:
                    str_to_hyp[hyp_str] = (expanded, len(new_hypotheses))
                    new_hypotheses.append(expanded)

        new_scores = np.array([score_hypothesis(h, weights, states_cnt) 
                                        for h in new_hypotheses])
        best_hyp_indices = np.argpartition(
            -new_scores, beam_width)[:beam_width]

        hypotheses = [new_hypotheses[i] for i in best_hyp_indices]

    best = max(hypotheses, key=lambda h: score_hypothesis(h, weights, states_cnt))
    return best
