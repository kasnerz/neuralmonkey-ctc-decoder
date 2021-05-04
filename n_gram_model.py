import kenlm
import copy
from hypothesis import Hypothesis, ExpandFunction


class NGramModel(object):

    def __init__(self, path: str):
        self.model = kenlm.LanguageModel(path)

    def expand_token(
            self, prev: Hypothesis, token: str,
            token_score: float) -> Hypothesis:

        if prev.lm_state is None:
            prev_state = kenlm.State()
            self.model.BeginSentenceWrite(prev_state)
        else:
            prev_state = prev.lm_state

        new_lm_state = kenlm.State()
        token_lm_score = self.model.BaseScore(prev_state, token, new_lm_state)

        hyp = copy.deepcopy(prev)
        hyp.expand_by_token(token, token_score, token_lm_score, new_lm_state)

        return hyp