import torch
import numpy as np

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Noise:
    @staticmethod
    def uniform_noise(batch_size, dim, bound):
        return Tensor(np.random.uniform(low= -bound, high= bound, size= (batch_size, dim, 1, 1)))

    @staticmethod
    def normal_noise(batch_size, dim, std):
        return Tensor(np.random.normal(loc= 0, scale= std, size= (batch_size, dim, 1, 1)))
