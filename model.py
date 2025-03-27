import torch as th
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import get_laplacian

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Linear
from ChebnetII_pro import ChebnetII_prop
import pywt


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(hid_dim, 64)
        #self.fc2 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, n_classes)
        #self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc1(x)
        ret = self.fc2(ret)
        ret = self.fc3(ret)
        #ret = self.fc4(ret)
        #ret = self.fc(x)
        return ret

        
class Discriminator1(nn.Module):
    def __init__(self, dim):
        super(Discriminator1, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)


    def forward(self, h1, h2, h3, h4, c):
        c_x = c.expand_as(h1).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x).squeeze(1)
        sc_2 = self.fn(h1, c_x).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x).squeeze(1)
        sc_4 = self.fn(h3, c_x).squeeze(1)

        #print(sc_1.shape, sc_2.shape, sc_3.shape, sc_4.shape)
        logits = th.cat((sc_1, sc_2, sc_3, sc_4))

        return logits

class Discriminator2(nn.Module):
    def __init__(self, dim):
        super(Discriminator2, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, s0_ll, s1_ll, h2_mean):
        c_s0 = h2_mean.expand_as(s0_ll).contiguous()
        c_s1 = h2_mean.expand_as(s1_ll).contiguous()

        # positive
        sc_1 = self.fn(s0_ll, c_s0).squeeze(1)

        # negative
        sc_2 = self.fn(s1_ll, c_s1).squeeze(1)

        #print(sc_1.shape, sc_2.shape, sc_3.shape, sc_4.shape)
        logits = th.cat((sc_1, sc_2))
        #print('out2',logits.shape)
        return logits


def compute_laplacian(edge_index, num_nodes):
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight=None,  normalization='sym', num_nodes=num_nodes)  
    return edge_index, edge_weight  


import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import pywt

class WaveletTransform(nn.Module):
    def __init__(self, in_dim, out_dim, wavelet='sym4', level=4, threshold_value=10):
        super(WaveletTransform, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.wavelet = wavelet  
        self.level = level  
        self.threshold_value = threshold_value 

    def forward(self, x, edge_index, num_nodes):
        x_transformed_denoised = self.wavelet_denoising(x)

        return x_transformed_denoised

    def wavelet_denoising(self, data):
        # Detach from the computation graph and move to CPU
        data_cpu = data.detach().cpu().numpy()
        
        # Use PyWavelets for wavelet decomposition and thresholding
        coeff = pywt.wavedec(data_cpu, self.wavelet, level=self.level)
        
        # Apply thresholding to the high-frequency parts
        #coeff[1:] = [pywt.threshold(i, value=self.threshold_value, mode='soft') for i in coeff[1:]]
        
        # Apply thresholding to the high-frequency parts (remove large coefficients)
        coeff[1:] = [i * (abs(i) <= self.threshold_value) for i in coeff[1:]]
        
        # Reconstruct the signal using the inverse wavelet transform
        denoised_data = pywt.waverec(coeff, self.wavelet)
        
        # Convert the denoised data back to a tensor and return it
        return torch.tensor(denoised_data, device=data.device)  # Ensure it's on the correct device

def get_wav(in_channels,phi, pool=True):
    haar_low_pass = np.array([phi / np.sqrt(2),  phi / np.sqrt(2)])
    haar_high_pass = np.array([phi / np.sqrt(2), -phi / np.sqrt(2)])

    filter_low = torch.from_numpy(haar_low_pass).unsqueeze(0).unsqueeze(0)  
    filter_high = torch.from_numpy(haar_high_pass).unsqueeze(0).unsqueeze(0)  

    if pool:
        net = nn.Conv1d  
    else:
        net = nn.ConvTranspose1d 

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0,groups=in_channels, bias=False)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0,groups=in_channels,  bias=False)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False

    LL.weight.data = filter_low.float().expand(in_channels, -1, -1).clone()
    LH.weight.data = filter_high.float().expand(in_channels, -1, -1).clone()

    return LL, LH



class WavePool(nn.Module):
    def __init__(self, in_channels,phi):
        super(WavePool, self).__init__()
        self.LL, self.LH = get_wav(in_channels,phi)

    def forward(self, x):

        ll, lh = self.LL(x), self.LH(x)
        return ll, lh



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class WaveLayer(nn.Module):
    def __init__(self, d_model,phi,dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.wavepool_1 = WavePool(1,phi)
        self.wavepool_2 = WavePool(1,phi)

        self.instance_norm = nn.InstanceNorm1d(256)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.linear = nn.Linear(64, 128)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    ### HiLo Wavelet Self-Attention
    def forward_post(self, tgt2):

        ll_2, lh_2 = self.wavepool_2(tgt2.unsqueeze(1))

        ll_2_flat = ll_2.squeeze(1)
        lh_2_flat = lh_2.squeeze(1)

        ll_2 = self.linear(ll_2_flat)
        lh_2 = self.linear(lh_2_flat)

        return ll_2, lh_2

    def forward(self, tgt2):
        return self.forward_post(tgt2)



class Model(nn.Module):
    def __init__(self, in_dim, out_dim, K, dprate, dropout, is_bns, act_fn,n_node,phi):
        super(Model, self).__init__()

        self.encoder = ChebNetII(num_features=in_dim, hidden=out_dim, K=K, dprate=dprate, dropout=dropout, is_bns=is_bns, act_fn=act_fn)

        #self.wavelet_transform = WaveletTransform(128, 128)
        self.wave_layer1 = WaveLayer(out_dim,phi)
        self.wave_layer2 = WaveLayer(out_dim,phi)

        self.disc1 = Discriminator1(out_dim)
        self.disc2 = Discriminator2(out_dim)
        self.disc3 = Discriminator2(out_dim)

        self.act_fn = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def get_embedding(self, edge_index, feat,n_node):
        h1 = self.encoder(x=feat, edge_index=edge_index, highpass=True)  #high-pass
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)    #low-pass

        h2_ll,h2_lh=self.wave_layer1(h2)
        h1_hl,h1_hh=self.wave_layer1(h1) 

        h = torch.mul(self.alpha, h2_lh) + torch.mul(self.beta, h1_hh)

        return h.detach()

    def forward(self, edge_index, feat, feat_s0, feat_s1, edge_index_s0, edge_index_s1, shuf_feat,n_node):
        # positive
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)    #low-pass
        h1 = self.encoder(x=feat, edge_index=edge_index, highpass=True)     #high-pass

        h2_ll,h2_lh=self.wave_layer1(h2) #low
        h1_hl,h1_hh=self.wave_layer2(h1) #high

        h2_mean = self.act_fn(torch.mean(h2, dim=0))
        h1_mean = self.act_fn(torch.mean(h1, dim=0))

        h = torch.mul(self.alpha, h2_lh) + torch.mul(self.beta, h1_hh)
        c = self.act_fn(torch.mean(h, dim=0))

        # negative
        h4 = self.encoder(x=shuf_feat, edge_index=edge_index, highpass=False) #low-pass
        h3 = self.encoder(x=shuf_feat, edge_index=edge_index, highpass=True)  #high-pass

        h4_ll,h4_lh=self.wave_layer1(h4)  #low-pass
        h3_ll,h3_lh=self.wave_layer2(h3)  #high-pass


        #GCL
        h2_s0=self.encoder(x=feat_s0, edge_index=edge_index_s0, highpass=False) #low-pass
        h2_s1=self.encoder(x=feat_s1, edge_index=edge_index_s1, highpass=False)

        h1_s0=self.encoder(x=feat_s0, edge_index=edge_index_s0, highpass=True) #high-pass
        h1_s1=self.encoder(x=feat_s1, edge_index=edge_index_s1, highpass=True)

        s0_ll,s0_lh=self.wave_layer1(h2_s0)
        s1_ll,s1_lh=self.wave_layer2(h2_s1)

        s0_hl,s0_hh=self.wave_layer1(h1_s0)
        s1_hl,s1_hh=self.wave_layer2(h1_s1)

        #out = self.disc(h1_wavelet, h2_wavelet, h3_wavelet, h4_wavelet, c)
        out1 = self.disc1(h1, h2_lh, h3, h4_lh, c) #loss_p
        out2 = self.disc2(s0_ll,s1_ll,h2_mean) ##loss_h for low-pass
        out3 = self.disc3(s0_hl,s1_hl,h1_mean) ##loss_h for high-pass

        return out1,out2,out3

    def get_predictive(self, edge_index, feat,n_node):
        h2 = self.encoder(x=feat, edge_index=edge_index, highpass=False)    #low-pass

        h2_ll,h2_lh=self.wave_layer1(h2)

        return h2_ll.detach()


class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=False, act_fn='relu'):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.prop1 = ChebnetII_prop(K=K)
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, highpass=True):

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, highpass=highpass)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, highpass=highpass)


        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x

class SenClsf(nn.Module):
    def __init__(self, emb_dim, args):
        super(SenClsf, self).__init__()
        self.classifierSen = nn.Linear(emb_dim, 1)  
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()  

    def forward(self, z):
        s = self.classifierSen(z) 
        return s 
    #def sensitive_pred(self, z):
        #return self.classifierSen(z)

    def loss(self, origin_feature, pred_s):
        return self.criterion(pred_s, origin_feature) 
