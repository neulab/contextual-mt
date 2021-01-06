from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import nn

@register_criterion("attention_loss")
class AttentionLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, head=None):
        """Compute the loss for the given sample.
        """

        _, output_features = model(**sample["net_input"]) # output_features = {"attn": [attn], "inner_states": inner_states}
        highlights = sample["highlights"]

        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output