import torch
from .base import BaseQuantizer
from pruner.fisher import compute_fisher_info, compute_fisher_info_taylor1

class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type, use_fisher_hessian=False):
        super(Log2Quantizer, self).__init__(
            bit_type,
            observer,
            module_type,
        )
        self.softmax_mask = None
        self.use_fisher_hessian = use_fisher_hessian

    def quant(self, inputs):
        if self.use_fisher_hessian:
            fisher_info = self.get_fisher_info(inputs)
            hessian_info = self.get_hessian_info(inputs)
            rounds = torch.round(-1 * inputs.log2() * fisher_info * hessian_info)
            print("Update the quantization interval using log2's use_fisher_hessian...")
        else:
            rounds = torch.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def get_fisher_info(self, data):
        return compute_fisher_info(data)

    def get_hessian_info(self, data):
        return compute_fisher_info_taylor1(data)

    def dequantize(self, inputs):
        outputs = 2**(-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs
