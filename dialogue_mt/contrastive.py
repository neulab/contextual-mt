import logging
import math
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.sequence_scorer import SequenceScorer
from fairseq.logging import progress_bar

logger = logging.getLogger()


def main():
    # Parse command-line arguments for generation
    parser = options.get_generation_parser()
    parser.add_argument(
        "--contra",
        default="/projects/tir4/users/kayoy/dialogue-translation/OpenSubs/one",
    )
    parser.add_argument(
        "--output",
        default="/projects/tir4/users/kayoy/dialogue-translation/dialogue-mt/contra/pred.txt",
    )
    args = options.parse_args_and_arch(parser)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task
    task = tasks.setup_task(args)
    task.load_contra()

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset("contra"),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions()]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    scorer = SequenceScorer(task.target_dictionary)

    scores = []
    ids = []
    tgts = []
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        tgts.append(tgt_dict.string(sample["target"]))
        if "net_input" not in sample:
            continue

        hypos = scorer.generate([model], sample)
        for i, sample_id in enumerate(sample["id"].tolist()):
            ids.append(sample_id)
            hypo = hypos[i][0]
            scores.append(hypo["score"] / math.log(2))

    scores = [x for _, x in sorted(zip(ids, scores))]

    with open(args.output, "w") as file:
        for i, s in zip(ids, scores):
            file.write(f"{s}\n")

    print("done!")


if __name__ == "__main__":
    main()
