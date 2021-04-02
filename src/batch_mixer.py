import torch
import random
import numpy as np
from typing import List, Tuple

# non overlapping classes in freq domain
NOC = {
    0: [1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 21, 22],
    1: [0, 2, 3, 12, 13, 15, 19, 22, 23],
    2: [0, 1, 5, 7, 8, 16, 18, 21, 22, 23],
    3: [0, 1, 5, 7, 8, 16, 18, 21, 22, 23],
    4: [0, 5, 7, 13, 15, 22, 23],
    5: [2, 3, 4, 6, 12, 13, 15, 16, 19, 21],
    6: [0, 5, 7, 22, 23],
    7: [2, 3, 4, 6, 12, 13, 15, 16, 19, 21],
    8: [0, 2, 3, 12, 13, 15, 19, 22, 23],
    9: [0, 22, 23],
    10: [],
    11: [0, 13, 15, 22, 23],
    12: [0, 1, 5, 7, 8, 22, 23],
    13: [0, 1, 4, 5, 7, 8, 11, 14, 16, 18, 21, 22, 23],
    14: [13, 15, 22],
    15: [0, 1, 4, 5, 7, 8, 11, 14, 16, 17, 18, 20, 21, 22, 23],
    16: [0, 2, 3, 5, 7, 13, 15, 19, 22, 23],
    17: [15, 22],
    18: [0, 2, 3, 13, 15, 19, 22, 23],
    19: [0, 1, 5, 7, 8, 16, 18, 21, 22, 23],
    20: [15, 22, 23],
    21: [0, 2, 3, 5, 7, 13, 15, 19, 22, 23],
    22: [0, 1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    23: [1, 2, 3, 4, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 21]
}


class BatchMixer:
    
    def __init__(self, noc: dict = NOC, p: float = 0.95, merge_mode=torch.min):
        """Mix batch of spectrograms, using information about classes 
        intersection in frequency domain
        
        Args:
            noc (dict): non-overlapping by frequency classes dict
            p (float): mixup probability
        """
        
        self.noc = noc
        self.p = p
        self.merge_mode = merge_mode
    
    def get_non_overlapping_classes(self, labels: list): # e.g. lables = [1, 4, 5]
        sets = [set(self.noc[label]) for label in labels]
        result = list(set.intersection(*sets)) if sets else []
        return result
    
    def get_candidates(self, src_labels: List[int], target_labels: List[List[int]]) -> List[int]:
        """Return candidate indexes for mixup"""
        possible_labels = set(self.get_non_overlapping_classes(src_labels))
        indexes = []
        for i, target_labels_i in enumerate(target_labels):
            if set(target_labels_i).issubset(possible_labels):
                indexes.append(i)
        return indexes
    
    def mixup(self, image_1, image_2, labels_1, labels_2):
        image = self.merge_mode(image_1, image_2)
        labels = torch.clamp(labels_1 + labels_2, 0, 1)
        return image, labels
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # convert [0, 1, 0, 1] to [1, 3] representation
        idx_labels = [list(np.where(l)[0]) for l in labels.cpu().numpy()]
        
        new_images, new_labels = [], []
        
        for i, image_i in enumerate(images):
            src_labels = idx_labels[i]
            candidates = self.get_candidates(src_labels, idx_labels)
            
            if random.random() < self.p and candidates:
                j = random.choice(candidates)
                image_j = images[j]
                image, label = self.mixup(image_i, image_j, labels[i], labels[j])
            else:
                image, label = image_i, labels[i]
                
            new_images.append(image)
            new_labels.append(label)
            
        return torch.stack(new_images, dim=0), torch.stack(new_labels, dim=0)