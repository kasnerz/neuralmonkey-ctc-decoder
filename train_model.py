#!/usr/bin/env python3

# pylint: disable=unused-import, wrong-import-order
import sys

sys.path.append("./neuralmonkey")
import neuralmonkey.checkpython
# pylint: enable=unused-import, wrong-import-order

import argparse
import json
import os
import numpy as np

from neuralmonkey.config.configuration import Configuration
from neuralmonkey.experiment import Experiment
from neuralmonkey.logging import log
from neuralmonkey.decoders import CTCDecoder
from neuralmonkey.runners import PlainRunner
from neuralmonkey.runners.tensor_runner import RepresentationRunner
from neuralmonkey.dataset import BatchingScheme

from train_beam_search import train_weights
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
    parser.add_argument("--kenlm", type=str,
                        help="Path to a KenLM model arpa file.")
    parser.add_argument("--prefix", type=str,
                        help="Path used as a prefix of stored checkpoints.")
    parser.add_argument("--lm-weight", type=float,
                        help="Default weight of the language model.")
    parser.add_argument("--null-trail-weight", type=float,
                        help="Default weight of the null-trailing feature.")
    parser.add_argument("--nt-ratio-weight", type=float,
                        help="Default weight of the null-token ratio feature.")

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

    weights = {}

    if args.lm_weight is not None:
        weights['lm_score'] = args.lm_weight

    if args.null_trail_weight is not None:
        weights['null_trailing'] = args.null_trail_weight

    if args.nt_ratio_weight is not None:
        weights['null_token_ratio'] = args.nt_ratio_weight

    if not weights:
        raise ValueError("No default weights specified, nothing to train.")


    ctc_decoder = None
    for runner in exp.model.runners:
        if (isinstance(runner, PlainRunner)
                and isinstance(runner.decoder, CTCDecoder)):
            ctc_decoder = runner.decoder
            break

    if ctc_decoder is None:
        raise ValueError(
            "Was not able to detect CTC decoder in the configuration.")

    print("Loading language model")
    lm = NGramModel(args.kenlm)
    print("LM loaded")

    logits_runner = RepresentationRunner(
        output_series="logits", encoder=ctc_decoder, attribute="logits")
    exp.model.runners = [logits_runner]


    dataset = datasets_model.test_datasets[0]
    singleton_batches = dataset.batches(BatchingScheme(1))

    DATASET_SIZE = dataset.length
    CHECKPOINTS = 5
    CHECKPOINT_ITERS = int(DATASET_SIZE / CHECKPOINTS)

    print("{} sentences in the dataset, checkpoint every {} sentences ({} checkpoints in total).".format(
        DATASET_SIZE, CHECKPOINT_ITERS, CHECKPOINTS))

    for i, sent_dataset in enumerate(singleton_batches):
        ctc_model_result = exp.run_model(
            sent_dataset, write_out=False, batch_size=1)

        logits = np.squeeze(ctc_model_result[1]['logits'],
            axis=1)
        target = ctc_model_result[2]['target'][0]

        train_weights(logits, args.beam, ctc_decoder.vocabulary,
            target, weights, lm)

        print("[{}] Weights:".format(i+1),
              ", ".join(
                ["{}: {:.3f}".format(key, value) for key, value in weights.items()]))

        if i != 0 and (i+1) % CHECKPOINT_ITERS == 0:
            with open("{}.{}".format(args.prefix, int(i/CHECKPOINT_ITERS)), "w") as f:
                for key, value in weights.items():
                    f.write("{}={:.3f}\n".format(key.upper(), value))

            print("\nCheckpoint saved.\n")


    for session in exp.config.model.tf_manager.sessions:
        session.close()


if __name__ == "__main__":
    main()
