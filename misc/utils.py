from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import tensorflow as tf
import tempfile
from six.moves import cPickle as pickle


def torch_save(obj, path):
    local_path = tempfile.NamedTemporaryFile(delete=False).name
    torch.save(obj.cpu(), local_path)
    tf.gfile.Copy(local_path, path, overwrite=True)
    os.remove(local_path)


def torch_load(path):
    local_path = tempfile.NamedTemporaryFile(delete=False).name
    tf.gfile.Copy(path, local_path, overwrite=True)
    obj = torch.load(local_path, map_location=torch.device('cpu'))
    os.remove(local_path)
    return obj


def pickle_save(obj, path, bytes=False):
    local_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(local_path, 'wb' if bytes else 'w') as f:
        pickle.dump(obj, f)
    tf.gfile.Copy(local_path, path, overwrite=True)
    os.remove(local_path)


def pickle_load(path, bytes=False):
    local_path = tempfile.NamedTemporaryFile(delete=False).name
    tf.gfile.Copy(path, local_path, overwrite=True)
    with open(local_path, 'rb' if bytes else 'r') as f:
        obj = pickle.load(f)
    os.remove(local_path)
    return obj


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True


def decode_sequence(ix_to_word, seq):
    # Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
