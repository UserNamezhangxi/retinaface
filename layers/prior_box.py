from math import ceil
from itertools import product as product

import torch


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    start_kx = min_size / self.image_size[1]
                    start_ky = min_size / self.image_size[0]

                    center_x = (j + 0.5) * self.steps[k] / self.image_size[1]
                    center_y = (i + 0.5) * self.steps[k] / self.image_size[0]

                    anchors += [center_x, center_y, start_kx, start_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
