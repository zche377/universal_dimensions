"""
    Adapted code from Raj Magesh
"""        
from typing import Protocol
from abc import abstractmethod

from PIL import Image
import pandas as pd



class StimulusSet(Protocol):
    identifier: str
    metadata: pd.DataFrame

    @abstractmethod
    def __getitem__(self, stimulus_id: str) -> Image.Image:
        pass