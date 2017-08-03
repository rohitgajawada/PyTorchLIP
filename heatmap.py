import torch
from torch.autograd import Variable
import torch.nn as nn

class Heatmap():
    def __init__(self, opt):
        self.opt = opt

    def forward(self, input, truth):
        input = input[:, :, 0]
        classes = []
        for i in range(opt.nclasses):
            temp = (input == i)
            classes.append(temp)

        centers = self.calculate_center(temp) 

    def calculate_center(input):


    def gaussianmap(input):
        var = self.opt.variance
