from xmlrpc.client import boolean
import requests
import shutil
import hashlib
import os
from pathlib import Path


# Download maps for model codes
DOWNLOAD_MAP = {
    'big': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/big.tar.gz',
    'small': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/ja-en/small.tar.gz',
    'spm': 'https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/spm_models/en-ja_spm.tar.gz'
}

# Model name to SHA 256 hases for file
DOWNLOAD_INTEGRITY_CHECK = {
    'big': '7517753B6FEB8594D3C86AD7742DBC49203115ADD21E8A6C7542AA2AC0DF1C6A',
    'small': '',
    'spm': '12EE719799022B9EF102CE828209E53876112B52B4363DC277CACA682B1B1D2E'
}

EXTRACTED_MODEL_MAPPINGS = {
    'big': {
        '/big/big.pretrain.pt': '/checkpoints/big.pretrain.pt',
        '/big/dict.en.txt': '/data-bin/dict.en.txt',
        '/big/dict.ja.txt': '/data-bin/dict.ja.txt'
    },
    'small': '',
    'spm': {
        '/enja_spm_models/spm.ja.nopretok.model': '/spe/spm.ja.nopretok.model',
        '/enja_spm_models/spm.en.nopretok.model': '/spe/spm.en.nopretok.model'
    }
}

class JParaCrawlTranslator:
    def __init__(self, model_size = 'small', model_path = None):
        if not model_size in ['small', 'big']:
            raise ValueError(f"model_size passed as {model_size} should be one of 'small' or 'big'")

        self.model_size = model_size
        self.base_model_path = model_path if model_path != None else self._get_default_model_base_path()
        
    def supports(self, src, tgt):
        return src == 'ja' and tgt == 'eng'

    

    # Gets a path a model zip file should be
    def download(self) -> bool:
        # Download
        if not self.is_model_downloaded('spm'):
            self._download_model('spm')
            if not self._verify_model('spm'):
                return False
                
        
        if not self.is_model_downloaded(self.model_size):
            self._download_model(self.model_size)
            self._verify_model(self.model_size)
        
        # Extract
        if not self.is_model_extracted('spm'):
            self._extract_model('spm')

        if not self.is_model_extracted(self.model_size):
            self._extract_model(self.model_size)
        
        # Setup (move files)
        if not self.is_model_setup('spm'):
            self.setup_model('spm')
        
        if not self.is_model_setup(self.model_size):
            self.setup_model(self.model_size)

    # Gets if model has been downloaded (Will count the model already being loaded OR extracted as the model being downloaded)
    def is_model_downloaded(self, model):
        return (self.is_model_setup(model)
         or self.is_model_extracted(model) 
         or os.path.exists(self._get_model_zip_path(model)))
    
    # Gets if model has been extracted
    def is_model_extracted(self, model):
        pass

    # Gets if model has been setup
    def is_model_setup(self, model):
        return os.path.exists(self._get_model_setup_path(model))
    

    def _get_model_zip_path(self, model):
        return os.path.join(self.base_model_path, f"{model}/download/{model}.tar.gz")
    
    def _get_model_extracted_path(self, model):
        return os.path.join(self.base_model_path, f"{model}/extract/{model}/")

    def _get_model_setup_path(self, model):
        return os.path.join(self.base_model_path, f"{model}/setup/")
    
    def _get_default_model_base_path(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.abspath(os.path.join(file_dir, "../../models/jparacrawl"))


    


    # Downloads a model
    def _download_model(self, model):
        download_path = self._get_model_zip_path(model)
        download_dir = os.path.dirname(download_path)
        download_url = DOWNLOAD_MAP[model]
        print(f"Downloading model {model} to: {download_path}. (This could take a while...)") 
        
        # Make sure download dir is there and then begin pulling the file
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        self._download_file(download_path, download_url)

    # Verifies a Model 
    def _verify_model(self, model):
        zip_file = self._get_model_zip_path(model)

        print(f"Verifying intergity of downloaded file {zip_file}")
        sha256_calculated = self._get_digest(model)
        sha256_pre_calculated = DOWNLOAD_INTEGRITY_CHECK[model]

        if sha256_calculated != sha256_pre_calculated:
            print(f"Mismatch behind downloaded and created hashes {sha256_calculated} ")
            return False
        return True

    # Extract
    def _extract_model(self, model):
        zip_file = self._get_model_zip_path(model)
        target_path = self._get_model_extracted_path(model)
        shutil.unpack_archive(zip_file, target_path)

    # Downloads a file
    def _download_file(url, file_path):
        with requests.get(url, stream=True) as r:
            with open(file_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

    # Hashes a file in a stream
    def _get_digest(file_path):
        h = hashlib.sha256()
        BUF_SIZE = 65536 

        with open(file_path, 'rb') as file:
            while True:
                # Reading is buffered, so we can read smaller chunks.
                chunk = file.read(BUF_SIZE)
                if not chunk:
                    break
                h.update(chunk)

        return h.hexdigest()

