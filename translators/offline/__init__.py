from .nllb import NLLBTranslator
from .chain_translator import ChainTranslator

def get_offline_translator():
    return ChainTranslator([
        NLLBTranslator('big')
    ])