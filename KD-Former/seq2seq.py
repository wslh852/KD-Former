import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from enc_dec import Encoder_Decoder
import torch_utils

class EncodeDecodeModel(object):
    def __init__(self, source_seq_len, target_seq_len, num_joint, in_chans, embed_dim_ratio, depth, num_heads, qkv_bias, loss_type, mlp_ratio, qk_scale, drop_rate, attn_drop_rate, drop_path_rate):
        super(EncodeDecodeModel, self).__init__()
        self.epoch = 0
        self.model = Encoder_Decoder(source_seq_len, target_seq_len, num_joint, embed_dim_ratio, in_chans, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate).cuda()

        self.input_dim=num_joint*in_chans


    def scheduler_step(self):
        self.epoch = self.epoch + 1
        self.model_scheduler.step()

   
   

    def eval(self, input, target):
        self.model.eval()

        outputs = self.model(input, target)

        return outputs

if __name__ == '__main__':
    pass
