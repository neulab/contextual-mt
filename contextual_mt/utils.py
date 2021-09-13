import torch


def encode(s, spm, vocab):
    """binarizes a sentence according to sentencepiece model and a vocab"""
    tokenized = " ".join(spm.encode(s, out_type=str))
    return vocab.encode_line(tokenized, append_eos=False, add_if_not_exist=False)


def decode(ids, spm, vocab):
    """decodes ids into a sentence"""
    tokenized = vocab.string(ids)
    return spm.decode(tokenized.split())


def create_context(sentences, context_size, break_id=None, eos_id=None):
    """based on list of context sentences tensors, creates a context tensor"""
    context = []
    # TODO: check if there is a bug here when context_size > len(sentences)
    for s in sentences[max(len(sentences) - context_size, 0) :]:
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
