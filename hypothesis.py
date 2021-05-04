from typing import Any, List, NamedTuple, Callable
import copy
import numpy as np

class Hypothesis:
    def __init__(self,
        tokens : List[str],
        ctc_score : float, # log probability from the CTC model
        lm_score : float,  # log probability from the LM
        lm_state : Any):

        self.tokens = tokens
        self.ctc_score = ctc_score
        self.lm_score = lm_score
        self.lm_state = lm_state
        self.null_total = 0
        self.null_trailing = 0

    def recombine_with(self, hyp):
        self.ctc_score = np.logaddexp(self.ctc_score, hyp.ctc_score)
        self.null_trailing = max(hyp.null_trailing, self.null_trailing)

    def expand_by_null(self, null_score: float):
        self.ctc_score += null_score
        self.null_total += 1
        self.null_trailing += 1

    def expand_by_token(self, token: str, token_score: float, 
            token_lm_score: float, new_lm_state: Any):

        self.tokens.append(token)
        self.ctc_score += token_score
        self.lm_score += token_lm_score
        self.lm_state = new_lm_state
        self.null_trailing = 0

    def __repr__(self):
        return "Hypothesis(tokens={}, ctc_score={}, lm_score={}, null_total={}, null_trailing={})".format(
            self.tokens, 
            self.ctc_score, 
            self.lm_score,
            self.null_total,
            self.null_trailing)

    def __deepcopy__(self, memo=None):
        hyp = Hypothesis(
            self.tokens[:],
            self.ctc_score,
            self.lm_score,
            self.lm_state)

        hyp.null_total = self.null_total
        hyp.null_trailing = self.null_trailing

        return hyp

ExpandFunction = Callable[[Hypothesis, str, float], Hypothesis]
