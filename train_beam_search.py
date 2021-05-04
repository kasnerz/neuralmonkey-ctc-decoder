#!/usr/bin/env python3

from typing import List
import numpy as np
import copy
import pprint as pp
from scipy.misc import logsumexp
from scipy.stats import beta

from neuralmonkey.vocabulary import Vocabulary
from n_gram_model import NGramModel
from hypothesis import Hypothesis, ExpandFunction


from beam_search import score_hypothesis, compute_feature, \
                        log_softmax, expand_null, empty_hypothesis

def list_startswith(list1, list2):
    return all([token1 == token2 for token1, token2 in zip(list1, list2)])

def update_weights(violation_hyp: Hypothesis, target_hyp: Hypothesis, 
                        weights: dict, states_cnt: int):
    LEARNING_RATE = 0.0005

    for key in weights.keys():
        weights[key] += LEARNING_RATE * (compute_feature(key, target_hyp, states_cnt) - 
                                         compute_feature(key, violation_hyp, states_cnt))


def add_expanded_hyp(
    ctc_table: np.ndarray,
    weights: dict,
    row: int,
    col: int, 
    candidate_hyp: Hypothesis,
    parent: (int, int)):

    current_hyp = ctc_table[row, col]

    weights = {
        "lm_score" : 0.0,
        "null_trailing" : 0.0,
        "null_token_ratio" : 0.0
    }

    if current_hyp:
        score_current = score_hypothesis(current_hyp[0], weights, 0)
        score_candidate = score_hypothesis(candidate_hyp, weights, 0)

        if score_candidate <= score_current:
            return

        candidate_hyp.recombine_with(current_hyp[0])

    ctc_table[row, col] = (candidate_hyp, parent)

def ctc_path(
    target: List,
    log_prob_table: np.ndarray,
    weights: dict,
    lm: NGramModel,
    vocabulary: Vocabulary) -> List[Hypothesis]:

    rows = len(target) + 1
    time_steps = len(log_prob_table)

    # error in data, target cannot be decoded
    if time_steps < len(target):
        return None

    ctc_table = np.empty(shape=(rows, time_steps), dtype=tuple)

    # fill the starting cell with the empty hypothesis
    ctc_table[0,0] = (empty_hypothesis(), None)

    for time in range(time_steps-1):
        null_log_prob = log_prob_table[time, -1]

        # fill only the space around the diagonal
        min_row = max(0, rows - (time_steps-time))
        max_row = min(time + 1, len(target))

        for row in range(min_row, max_row):
            hyp = ctc_table[row, time][0]
            next_token = target[row]
            next_token_idx = vocabulary.word_to_index[next_token]

            # add eps
            expanded = expand_null(hyp, null_log_prob)
            add_expanded_hyp(ctc_table, weights, row, time+1, 
                candidate_hyp=expanded, parent=(row, time))

            # add next token
            next_token_score = log_prob_table[time, next_token_idx]
            expanded = lm.expand_token(hyp, next_token, next_token_score)
            add_expanded_hyp(ctc_table, weights, row+1, time+1, 
                                candidate_hyp=expanded, parent=(row, time))

    # reconstruct path
    path = []
    hyp = ctc_table[rows-1, time_steps-1]

    # error in data
    if hyp is None:
        return None

    while True:
        path.append(hyp[0])
        prev_idx = hyp[1]

        if prev_idx is None:
            break

        hyp = ctc_table[prev_idx]

    path.reverse()
    return path


def train_weights(
        logits_table: np.ndarray,
        beam_width: int,
        vocabulary: Vocabulary,
        target: list,
        weights: dict,
        lm: NGramModel) -> List[str]:
    assert beam_width >= 1
    
    log_prob_table = log_softmax(logits_table)
    hypotheses = [empty_hypothesis()]
    time_steps = log_prob_table.shape[0]

    target_hyp_path = ctc_path(target, log_prob_table, weights, lm, vocabulary)

    # error in data
    if target_hyp_path is None:
        return

    states_cnt = len(log_prob_table)

    for time in range(len(log_prob_table)-1):
        log_probs = log_prob_table[time]

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

        for hyp_index, hyp in enumerate(hypotheses):
            for token_index, score in zip(best_tokens, best_scores):
                token = vocabulary.index_to_word[token_index]
                expanded = lm.expand_token(hyp, token, score)
                score = score_hypothesis(expanded, weights, states_cnt)
                hyp_str = " ".join(expanded.tokens)

                if hyp_str in str_to_hyp:
                    orig_hyp, hyp_index = str_to_hyp[hyp_str]
                    expanded.recombine_with(orig_hyp)
                    new_hypotheses[hyp_index] = expanded
                    str_to_hyp[hyp_str] = (expanded, hyp_index)
                else:
                    str_to_hyp[hyp_str] = (expanded, len(new_hypotheses))
                    new_hypotheses.append(expanded)

        target_candidates_indices = [i for i, h in enumerate(new_hypotheses) 
                                        if list_startswith(target, h.tokens)]
        new_scores = np.array([score_hypothesis(h, weights, states_cnt) 
                                for h in new_hypotheses])
        target_candidates = [new_hypotheses[i] 
                                for i in target_candidates_indices]
        target_candidates_tokens_cnt = np.array([len(h.tokens) 
                                for h in target_candidates])

        best_hyp_indices = np.argsort(-new_scores)
        target_hyp_ranks = np.in1d(best_hyp_indices, target_candidates_indices).nonzero()[0]

        hypotheses = [new_hypotheses[i] for i in best_hyp_indices[:beam_width]]

        # hypotheses are out of the beam or no hypotheses can be finished in time
        if (all(target_hyp_ranks >= beam_width) or                                      
            all(target_candidates_tokens_cnt + (time_steps - time) < len(target))):
                
                for i in range(beam_width):
                    violation_hyp = hypotheses[i]
                    target_hyp = target_hyp_path[time+1]

                    update_weights(violation_hyp, target_hyp, weights, states_cnt)

                return