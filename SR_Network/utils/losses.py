import torch

device = torch.device('cuda')

cri_gan1 = torch.nn.BCEWithLogitsLoss().to(device)
cri_fea = torch.nn.L1Loss().to(device)
cri_pix = torch.nn.L1Loss().to(device)

def cri_gan(inp, tf):
    op = torch.empty_like(inp).fill_(0.0)
    if tf:
        op = torch.empty_like(inp).fill_(1.0)      
    
    return cri_gan1(inp, op)