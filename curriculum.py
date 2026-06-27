import numpy as np
from torch.utils.data import Sampler


class CurriculumSampler(Sampler):

    def __init__(self, difficulties, max_epochs, p0=0.2, warmup_frac=0.5, seed=1):
        self.order = np.argsort(np.asarray(difficulties), kind="stable")
        self.n = len(self.order)
        self.max_epochs = max_epochs
        self.p0 = p0
        self.warmup_epochs = max(1, int(round(warmup_frac * max_epochs)))
        self.epoch = 0
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def current_pool_size(self):
        if self.epoch >= self.warmup_epochs:
            p = 1.0
        else:
            p = self.p0 + (1.0 - self.p0) * (self.epoch / self.warmup_epochs)
        return max(1, int(round(p * self.n)))

    def __iter__(self):
        k = self.current_pool_size()
        pool = self.order[:k].copy()
        self.rng.shuffle(pool)
        return iter(pool.tolist())

    def __len__(self):
        return self.current_pool_size()


def noise_alpha(epoch, max_epochs, warmup_frac=0.5, alpha0=0.0):
    warmup = max(1, int(round(warmup_frac * max_epochs)))
    if epoch >= warmup:
        return 1.0
    return alpha0 + (1.0 - alpha0) * (epoch / warmup)
