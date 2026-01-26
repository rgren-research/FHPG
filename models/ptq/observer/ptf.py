import torch
from .base import BaseObserver
from .utils import lp_loss
# Introducing PyTorch's loss function module
from torch.nn import MSELoss, L1Loss

'''

        This code defines a class named PtfObserver, which inherits from BaseObserver. 
        It primarily acts as an observer in Quantization Aware Training (QAT), 
        recording the maximum and minimum values of the data to help determine the quantization parameters.
'''
class PtfObserver(BaseObserver):
    # Call the initialization method of the base class BaseObserver, passing the same parameters.
    def __init__(self, module_type, bit_type, calibration_mode):
        super(PtfObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)
    # Receive a tensor v and reshape it.
    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    # The quantization parameters (scale and zero point) are calculated based on the recorded maximum and minimum values and the input tensors.
    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val
        # The upper and lower bounds qmax and qmin of the quantization range are defined.
        qmax = self.bit_type.upper_bound # The upper and lower limits of the quantization range are determined by bit_type.
        qmin = self.bit_type.lower_bound
        # Initialize best_score to a large number.
        # Calculate the difference between the maximum and minimum values, and calculate scale8 based on the quantization range.
        best_score = 1e+10
        # max_val_t = max_val.max()
        # min_val_t = min_val.min()
        #
        # scale8 = (max_val_t - min_val_t) / float(qmax - qmin)  # Formula (6). This allows for a better reflection of the channel's dynamic range.
        print("inputs Tensor shape:", inputs.shape)
        # Use the standardized standard deviation of the channels instead of max_val_t - min_val_t
        # std_dev = inputs.std(dim=(0, 2, 3), keepdim=True)  # Assume the input shape is NCHW
        std_dev = inputs.std(dim=(0, 1, 2), keepdim=True)
        scale8 = std_dev / float(qmax - qmin)
        scale8.clamp_(self.eps) # Limit the value of scale8 to a small positive number above self.eps to avoid division errors.

        # Use the channel mean instead of min_val_t
        # mean_val = inputs.mean(dim=(0, 2, 3), keepdim=True)
        mean_val = inputs.mean(dim=(0, 1, 2), keepdim=True)  # This ensures that the quantized signal maintains a distribution close to the original signal on each channel.
        zero_point = qmin - torch.round(mean_val / scale8)
        zero_point.clamp_(qmin, qmax)

        scale4 = scale8 / 2 # Calculate other scale factors: scale4, scale2, scale1
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        # Calculate the zero point

        # zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val) # Initialize a full sheet of shape scale_mask with the same shape as max_val.
        '''
        The input data is quantized using different quantization scales (scale1, scale2, scale4, scale8), 
        and the quantization error is calculated using the lp_loss function.
        '''
        for j in range(inputs.shape[2]): # Traversing the third dimension of the input tensor
            data = inputs[..., j].unsqueeze(-1)
            # The input data is quantized using scale1 and zero_point, and then dequantized to obtain data_q1.
            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            # Calculate the loss between the raw data and the quantized data.
            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            # Calculate the loss at all quantization scales and find the quantization scale corresponding to the minimum loss.
            score = [score1, score2, score4, score8]  #  The value of Alpha

            #  complexity term
            complexity = [torch.log2(scale1), torch.log2(scale2), torch.log2(scale4), torch.log2(scale8)]
            # lambda is a trade-off factor used to balance quantization error and complexity.
            lambda_factor = 0.1
            total_scores = [s + lambda_factor * c for s, c in zip(score, complexity)]

            # scale_mask[j] *= 2**score.index(min(score))
            # Update scale_mask
            scale_mask[j] *= 2 ** total_scores.index(min(total_scores))
        scale = scale1 * scale_mask # The final scale is the product of scale1 and scale_mask.
        return scale, zero_point  # Returns the final quantization scale and zero point.
'''
This class optimizes quantization performance by dynamically adjusting the quantization scale to find the quantization 
scheme with the least loss at different quantization levels.

The core function of this class is to collect statistical information during training and 
then calculate the optimal quantization parameters based on this information to quantize the model, 
thereby reducing model size and computational cost while maintaining a certain level of accuracy.
'''




