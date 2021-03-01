import torch.nn as nn
from . import blocks as B

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        in_nc, out_nc, nf, nb, gc = opt['in_nc'], opt['out_nc'], opt['nf'], opt['nb'], opt['gc']
        norm_type, act_type = opt['norm_type'], opt['act_type']
        mode = opt['mode']
        
        conv1 = B.conv_block(in_nc, nf, kernel_size=9, norm_type=None, act_type='relu')
        rb_blocks = [B.Residual_block(nf, kernel_size=3, gc=64, stride=1, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        conv2 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        conv3 = B.conv_block(nf, 3, kernel_size=33, norm_type=None, act_type=None, mode= mode)
        conv4 = B.conv_block(3, out_nc, kernel_size=3, norm_type=None, act_type=None, mode= mode)

        self.model = B.sequential(conv1, B.ShortcutBlock(B.sequential(*rb_blocks, conv2)), conv3, conv4)

    def forward(self, x):
        x = self.model(x)
        return x