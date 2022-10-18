import imp
from .base_offline_translator import BaseOfflineTranslator
from translators.offline.downloader.jparacrawldownloader import JParaCrawlDownloader

# Download maps for model codes
DOWNLOAD_MAP = {
    'big': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/big.tar.gz',
    'small': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/small.tar.gz',
    'spm': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/spm_models/en-ja_spm.tar.gz'
}

# Model name to SHA 256 hases for file
DOWNLOAD_INTEGRITY_CHECK = {
    'big': '7517753B6FEB8594D3C86AD7742DBC49203115ADD21E8A6C7542AA2AC0DF1C6A',
    'small': '7136FE12841C626B105A9E588F858A8E0B76E451B19839457D7473EC705D12B3',
    'spm': '12EE719799022B9EF102CE828209E53876112B52B4363DC277CACA682B1B1D2E'
}

# Extracted -> Setup Model Mappings
EXTRACTED_MODEL_MAPPINGS = {
    'big': {
        'big/big.pretrain.pt': 'checkpoints/big.pretrain.pt',
        'big/dict.en.txt': 'data-bin/big/dict.en.txt',
        'big/dict.ja.txt': 'data-bin/big/dict.ja.txt'
    },
    'small': {
        'small/small.pretrain.pt': 'checkpoints/small.pretrain.pt',
        'small/dict.en.txt': 'data-bin/small/dict.en.txt',
        'small/dict.ja.txt': 'data-bin/small/dict.ja.txt'
    },
    'spm': {
        'enja_spm_models/spm.ja.nopretok.model': 'spm/spm.ja.nopretok.model',
        'enja_spm_models/spm.en.nopretok.model': 'spm/spm.en.nopretok.model'
    }
}

# Downloader flow: download -> verify -> extract -> setup

class JParaCrawlTranslator(BaseOfflineTranslator):
    def __init__(self, model_size = 'small', model_path = None):
        if not model_size in ['small', 'big']:
            raise ValueError(f"model_size passed as {model_size} should be one of 'small' or 'big'")

        self.downloder = JParaCrawlDownloader(model_size=model_size)
        super.__init__()

    def supports(self, src, tgt):
        return src == 'ja' and tgt == 'en'

    def download(self) -> bool:
        return self.downloder.download()
    
    def load(self) -> bool:
        return 


