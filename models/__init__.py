from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .Att2inModel import Att2inModel
from .AttModel import (
    Att2in2Model, AdaAttModel, AdaAttMOModel, TopDownModel)


def setup(opt):
    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # FC model in self-critical
    elif opt.caption_model == 'fc':
        model = FCModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    else:
        raise Exception(
            "Caption model not supported: {}".format(opt.caption_model)
        )

    return model
