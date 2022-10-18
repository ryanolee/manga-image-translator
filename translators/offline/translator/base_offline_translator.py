
from abc import ABC, abstractmethod
import torch


class BaseOfflineTranslator(ABC):
    def __init__(self):
        super.__init__()
    
    """
    Checks if translator supports language direction
    """
    @abstractmethod
    def supports(self, src: str, tgt: str) -> bool:
        pass

    """
    Translates a scentence if a given language direction
    """
    @abstractmethod
    def translate_sentence(self, src: str, tgt: str) -> bool:
        pass

    """
    Download model to file
    """
    @abstractmethod
    def download(self) -> bool:
        pass
    
    """
    Loads model into memory
    """
    @abstractmethod
    def load(self) -> bool:
        pass
    
    """
    Loads model into memory
    """
    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    
    """
    Gets the total amount of GPU memory used by model 
    (Used so that model will be loaded into RAM if the GPU is likley to run out of VRAM)
    """
    def _get_used_gpu_memory(self) -> bool:
        return torch.cuda.mem_get_info()