from genericpath import exists
import requests
import sys
from tqdm import tqdm
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

class JParaCrawlDownloader:
    def __init__(self, model_size = 'small', model_path = None):
        if not model_size in ['small', 'big']:
            raise ValueError(f"model_size passed as {model_size} should be one of 'small' or 'big'")

        self.model_size = model_size
        self.base_model_path = model_path if model_path != None else self._get_default_model_base_path()
        
    def supports(self, src, tgt):
        return src == 'ja' and tgt == 'eng'

    # Downloads the model to the expected locations
    def download(self) -> bool:
        return self._download_extract_setup('spm') and self._download_extract_setup(self.model_size)

    # Runs the full Download -> Extract -> Setup pipeline for downloads
    def _download_extract_setup(self, model) -> bool:
        if not self.is_model_downloaded(model):
            self._download_model(model)
            if not self._verify_model(model):
                return False
        
        if not self.is_model_extracted(model):
            self._extract_model(model)
            self._download_model_cleanup(model)
        
        if not self.is_model_setup(model):
            self._setup_model(model)
            self._extract_model_cleanup(model)
        
        return True


    # Gets if model has been downloaded (Will count the model already being loaded OR extracted as the model being downloaded)
    def is_model_downloaded(self, model):
        return self.is_model_extracted(model) or os.path.exists(self._get_model_zip_path(model))
    
    # Gets if model has been extracted
    def is_model_extracted(self, model):
        # If the model is setup then it has been "Extracted"
        if self.is_model_setup(model):
            return True
        
        # Check for all files that were required from the extraction to state that is is complete otherwise
        extract_path = self._get_model_extracted_path(model)
        return all([os.path.exists(os.path.join(extract_path, tgt_path)) for tgt_path in EXTRACTED_MODEL_MAPPINGS[model].keys()])


    # Gets if model has been setup
    def is_model_setup(self, model):
        setup_path = self._get_model_setup_path()

        # Check that all target paths exist in the expected places
        return all([os.path.exists(os.path.join(setup_path, tgt_path)) for tgt_path in EXTRACTED_MODEL_MAPPINGS[model].values()])
    

    def _get_model_zip_path(self, model):
        return os.path.join(self.base_model_path, f"{model}/download/{model}.tar.gz")
    
    def _get_model_extracted_path(self, model):
        return os.path.join(self.base_model_path, f"{model}/extract/{model}/")

    def _get_model_setup_path(self):
        return os.path.join(self.base_model_path, "setup/")
    
    def _get_default_model_base_path(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.abspath(os.path.join(file_dir, "../../../models/jparacrawl"))

    # Downloads a model
    def _download_model(self, model):
        download_path = self._get_model_zip_path(model)
        download_dir = os.path.dirname(download_path)
        download_url = DOWNLOAD_MAP[model]
        print(f"Downloading: {download_url} (This could take a while...)") 
        
        # Make sure download dir is there and then begin pulling the file
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        self._download_file(download_url, download_path)

    def _download_model_cleanup(self, model):
        model_path = self._get_model_zip_path(model)
        print(f"Deleting: {model_path}")
        os.unlink(model_path)

    # Verifies a Model 
    def _verify_model(self, model):
        zip_file = self._get_model_zip_path(model)

        print(f"Verifying: {zip_file}")
        sha256_calculated = self._get_digest(zip_file)
        sha256_pre_calculated = DOWNLOAD_INTEGRITY_CHECK[model]

        if sha256_calculated.capitalize() != sha256_pre_calculated.capitalize():
            print(f"Verifying: Mismatch behind downloaded and created hashes {sha256_calculated} <-> {sha256_pre_calculated}")
            return False
        print("Verifying: OK!")
        return True

    # Extract
    def _extract_model(self, model):
        zip_file = self._get_model_zip_path(model)
        print(f"Extracting: {zip_file}")
        target_path = self._get_model_extracted_path(model)
        shutil.unpack_archive(filename=zip_file, extract_dir=target_path)
    
    def _extract_model_cleanup(self, model):
        extracted_path = self._get_model_extracted_path(model)
        print(f"Deleting: {extracted_path}")
        shutil.rmtree(extracted_path)

    # Setup
    def _setup_model(self, model):
        model_makeup = EXTRACTED_MODEL_MAPPINGS[model]
        extracted_path = self._get_model_extracted_path(model)
        setup_path = self._get_model_setup_path()
        for target_file, setup_destination in model_makeup.items():
            from_path = os.path.join(extracted_path, target_file)
            to_path =  os.path.join(setup_path, setup_destination)
            
            print(f"Moving: {from_path} -> {to_path}")
            # Create and move files
            Path(os.path.dirname(to_path)).mkdir(parents=True, exist_ok=True)
            shutil.move(from_path, to_path)
        

    # Downloads a file
    def _download_file(self, url, file_path):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        chunk_size = 1024
        with open(file_path, 'wb') as file, tqdm(
            desc=file_path,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            is_tty = sys.stdout.isatty()
            downloaded_chunks = 0
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

                # Fallback for non TTYs so output still shown
                downloaded_chunks += 1
                if not is_tty and downloaded_chunks % 1000 == 0:
                    print(bar)
                
                

    # Hashes a file in a stream
    def _get_digest(self, file_path):
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


