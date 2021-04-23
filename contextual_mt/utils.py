import sys

import torch
from fairseq import utils


def encode(s, spm, vocab):
    """ binarizes a sentence according to sentencepiece model and a vocab """
    tokenized = " ".join(spm.encode(s, out_type=str))
    return vocab.encode_line(tokenized, append_eos=False, add_if_not_exist=False)


def decode(ids, spm, vocab):
    """ decodes ids into a sentence """
    detokenized = vocab.string(ids)
    return spm.decode(detokenized.split())

def decode_scores(tokens, spm, vocab, token_scores, normalize=True):
    detokenized = vocab.string(tokens).split()
    word_scores = []
    for token, score in zip(detokenized, token_scores):
        if ('▁' in token) or ('_' in token):
            word_scores.append([score])
        else:
            word_scores[-1] += [score]
            #if token == ';':
            #    print(word_scores[-1])
    # if normalize:
    #     word_scores = [sum(w)/len(w) for w in word_scores]
    # else:
    #     word_scores = [sum(w) for w in word_scores]
    return word_scores

def token_to_word_cxmi(token_cxmis, documents, ids, tgt_spm):
    word_cxmis = []
    for k in range(len(token_cxmis)):
        tokens = tgt_spm.encode(documents[ids[k][0]][ids[k][1]][1], out_type=str)
        scores = token_cxmis[k]
        word_cxmi = []
        for token, score in zip(tokens, scores):
            if ('▁' in token) or ('_' in token):
                word_cxmi.append([score])
            else:
                word_cxmi[-1] += [score]
        word_cxmis.append(word_cxmi)
    return word_cxmis

def create_context(sentences, context_size, break_id=None, eos_id=None):
    """ based on list of context sentences tensors, creates a context tensor """
    context = []
    for s in sentences[len(sentences) - context_size :]:
        if context and break_id is not None:
            context.append(torch.tensor([break_id]))
        context.append(s)
    if eos_id is not None:
        context.append(torch.tensor([eos_id]))
    return torch.cat(context) if context else torch.tensor([]).long()


def parse_documents(srcs, refs, docids):
    # parse lines into list of documents
    documents = []
    prev_docid = None
    for src_l, tgt_l, idx in zip(srcs, refs, docids):
        if prev_docid != idx:
            documents.append([])
        prev_docid = idx
        documents[-1].append((src_l, tgt_l))
    return documents


class SequenceGapScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            target_probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            pred = torch.topk(probs, 5, dim=2)
            max_probs = pred.values[:,:,0].view(target_probs.size())
            return max_probs - target_probs, pred.indices

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                if is_single:
                    probs, pred = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs, pred = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                        "pred": pred[i],
                    }
                ]
            )
        return hypos
