import logging

logging.basicConfig(level=logging.INFO)

import torch

from bonner.computation.decomposition import PCA



class RankFilteredPCA(PCA):
    def __init__(
        self,
        n_components: int = None,
        filtered: bool = True,
        z_score: bool = False,
        **kwargs,
    ) -> None:
        self.filtered = filtered
        self.z_score = z_score
        self.source_std = None
        self.source_mean = None
        super().__init__(
            n_components=n_components,
            **kwargs,
        )

    def fit(self, x: torch.Tensor) -> None:
        if self.z_score:
            self.source_mean = x.mean(dim=-2, keepdim=True)
            self.source_std = x.std(dim=-2, keepdim=True)
            self.source_std[self.source_std == 0] = 1
            x = (x - self.source_mean) / self.source_std
        else:
            self.source_mean = torch.zeros(1)
            self.source_std = torch.ones(1)
        
        if self.n_components is not None or self.filtered:
            rank = torch.linalg.matrix_rank(x)
            self.n_components = int(rank)
            logging.info(f"Rank = {rank}")
            logging.info(f"Atol rank = {torch.linalg.matrix_rank(x, atol=1.0, rtol=0.0)}")
            
        super().fit(x)
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.source_mean) / self.source_std
        return super().transform(x)
    
    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        return super().inverse_transform(z) * self.source_std + self.source_mean