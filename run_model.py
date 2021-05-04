#!/usr/bin/env python3

# pylint: disable=unused-import, wrong-import-order
import sys

sys.path.append("./neuralmonkey")
import neuralmonkey.checkpython
# pylint: enable=unused-import, wrong-import-order

import argparse
import json
import os
import timeit

import numpy as np

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.experiment import Experiment
from neuralmonkey.logging import log
from neuralmonkey.decoders import CTCDecoder
from neuralmonkey.runners import PlainRunner
from neuralmonkey.runners.tensor_runner import RepresentationRunner
from neuralmonkey.dataset import BatchingScheme

from beam_search import decode_beam
from n_gram_model import NGramModel

def main() -> None:
    # pylint: disable=no-member,broad-except
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", metavar="INI-FILE",
                        help="the configuration file of the experiment")
    parser.add_argument("datasets", metavar="INI-FILE",
                        help="the configuration file of the experiment")
    parser.add_argument("--beam", metavar="BEAM_SIZE", type=int, default=10,
                        help="Beam size.")
    parser.add_argument("--kenlm", type=str, default=None,
                        help="Path to a KenLM model arpa file.")
    parser.add_argument("--lm-weight", type=float,
                        help="Weight of the language model.")
    parser.add_argument("--null-trail-weight", type=float,
                        help="Weight of the null-trailing feature.")
    parser.add_argument("--nt-ratio-weight", type=float,
                        help="Weight of the null-token ratio feature.")
    parser.add_argument("--out", type=str,
                        help="Path to the output file.")
    args = parser.parse_args()

    test_datasets = Configuration()
    test_datasets.add_argument("test_datasets")
    test_datasets.add_argument("batch_size", cond=lambda x: x > 0)
    test_datasets.add_argument("variables", cond=lambda x: isinstance(x, list))

    test_datasets.load_file(args.datasets)
    test_datasets.build_model()
    datasets_model = test_datasets.model

    exp = Experiment(config_path=args.config)
    exp.build_model()
    exp.load_variables(datasets_model.variables)

    ctc_decoder = None
    for runner in exp.model.runners:
        if (isinstance(runner, PlainRunner)
                and isinstance(runner.decoder, CTCDecoder)):
            ctc_decoder = runner.decoder
            break

    if ctc_decoder is None:
        raise ValueError(
            "Was not able to detect CTC decoder in the configuration.")

    logits_runner = RepresentationRunner(
        output_series="logits", encoder=ctc_decoder, attribute="logits")
    exp.model.runners = [logits_runner]

    dataset = datasets_model.test_datasets[0]
    singleton_batches = dataset.batches(BatchingScheme(1))
    print("Loading language model")
    lm = NGramModel(args.kenlm)
    print("LM loaded")

    weights = {}

    if args.lm_weight:
        weights['lm_score'] = args.lm_weight

    if args.null_trail_weight:
        weights['null_trailing'] = args.null_trail_weight

    if args.nt_ratio_weight:
        weights['null_token_ratio'] = args.nt_ratio_weight

    print("Weights:", weights)

    i = 0
    stats = []

    with open(args.out, 'w') as out_file:
        for sent_dataset in singleton_batches:

            t1 = timeit.default_timer()
            ctc_model_result = exp.run_model(
                sent_dataset, write_out=False, batch_size=1)
            t2 = timeit.default_timer()

            logits = np.squeeze(ctc_model_result[1]['logits'],
                axis=1)

            t3 = timeit.default_timer()
            best_hyp = decode_beam(
                logits, args.beam, ctc_decoder.vocabulary, lm=lm, weights=weights)
            t4 = timeit.default_timer()

            stats.append([len(best_hyp.tokens), t2-t1, t4-t3])

            output = " ".join([best_hyp.tokens][0])
            out_file.write(output + "\n")

            if i % 10 == 0:
                print("[{}] {}".format(i, output))
            i+=1

    with open(args.out + ".stats", 'w') as stats_file:
        for line in stats:
            stats_file.write("{} {:.3f} {:.3f}\n".format(*line))

    for session in exp.config.model.tf_manager.sessions:
        session.close()


if __name__ == "__main__":
    main()
