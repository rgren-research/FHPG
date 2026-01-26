from .base import BaseQuantizer

from pruner.fisher import compute_fisher_info, compute_fisher_info_taylor1
import torch
import numpy as np



# fisher，hessian
class UniformQuantizer(BaseQuantizer):
    #
    def __init__(self, bit_type, observer, module_type, use_fisher_hessian=True):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None

        self.use_fisher_hessian = use_fisher_hessian
        # self.module = None

    # def set_module(self, module):
    #     self.module = module

    # Update the quantization parameters scale and zero_point
    def update_quantization_params(self, data):
        # self.scale, self.zero_point = self.observer.get_quantization_params(
        #     *args, **kwargs)
        # if weights is None:
        #     weights = self.module.weight.data
        # weights = self.module.weight.data
        if self.use_fisher_hessian:
            fisher_info = self.get_fisher_info(data)
            hessian_info = self.get_compute_fisher_info_taylor1(data)
            self.scale, self.zero_point = self.adjust_quantization_params(fisher_info, hessian_info)
            print("Update the quantization interval using the use_fisher_hessian information of the uniform...")
            # self.scale, self.zero_point = self.adjust_quantization_params(fisher_info)
        else:  # If Fisher and Hessian information are not used, the default quantization interval update logic is used.
            self.scale, self.zero_point = self.observer.get_quantization_params(data)

    def get_fisher_info(self, data):
        # Methods for calculating Fisher information(Adjustments are needed)
        return compute_fisher_info(data)

    def get_compute_fisher_info_taylor1(self, data):
        # Methods for calculating Hessian information(Adjustments are needed)
        return compute_fisher_info_taylor1(data)

    def adjust_quantization_params(self, fisher_info, hessian_info):

        fisher_mean = fisher_info.mean()
        hessian_mean = hessian_info.mean()

        epsilon = 1e-8
        scale = hessian_mean / (fisher_mean + epsilon)

        zero_point = abs(fisher_mean - hessian_mean)
        zero_point = torch.tensor(zero_point).to(fisher_info.device)

        return scale, zero_point


    def quant(self, inputs, scale=None, zero_point=None):  # Perform quantification operations
        if scale is None:
            scale = self.scale
        if zero_point is None:  # If scale and zero_point are not provided, then member variables from the class are used.
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        # bit_type.lower_bound
        outputs = outputs.round().clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):  #Dequantization is achieved by subtracting zeros from the input and then multiplying by a scale.
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs