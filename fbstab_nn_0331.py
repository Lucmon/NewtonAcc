
import os
import sys
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn.init import constant_
import torch.optim as optim
import time
import numpy as np
import random
seed = 0
random.seed(seed) # Python random module.
from scipy.sparse import random
from easydict import EasyDict as edict

#from graphsage import GSSupervised
#from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup
from torch.nn import functional as F
from MultiHeadAttention import MultiheadAttention
from CoCoAttention import CoCoAttention
import torch.nn.init as init
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(seed) # 为CPU设置随机种子 
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子 
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU，为所有GPU设置随机种子 
np.random.seed(seed) # Numpy module. 
 
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True
#torch.autograd.set_detect_anomaly(True)

__C = edict()
cfg = __C
__C.TRAIN = edict()
__C.TRAIN.LR = 0.01 #g
__C.TRAIN.LR_DECAY = 0.1 #g
__C.TRAIN.LR_STEP = [10, 20] #g
__C.TRAIN.MOMENTUM = 0.9 #g
__C.logging = False
__C.cuda = True
__C.output = True
__C.train = False
__C._use_linformer = False #g
__C._att_dropout = 0.2 #g
__C.kdim = 8 # 32, 128, 256
__C.elementwise_affine = False
__C.double_attention = False
__C._use_gt_loss = False
__C._surrogate_func =True
__C._gumbel_softmax = True
__C._clamp_v = False
__C._proj_dropout = True
__C._newton_iters_prelearn = -1
__C._index_num_bound = 100
__C._duplicate_constraints = False
__C.smw_acc = False
__C.data_type = "svm"

# quadratic programming
class qp_struct:
    def __init__(self): 
        return

# parameters
m = 50
n = 10
sigma = 0.01
max_newton_iters = 500
max_prox_iters = 1
inner_tol = 1e-10
tol = 1e-6
rtol = 1e-12
dtol = 1e-12
alpha = 0.95
eps = 2.2e-16
inverse_time = 0
num_iters = 3
num_constraints = m*n

def nr(x,_lambda,v,qp):
    """
    calculate the residual of KKT system.
    """
    if cfg.data_type == "simulation":
        r1 = torch.matmul(qp.H, x) + torch.matmul(qp.G.t(),_lambda)\
            + torch.matmul(qp.A.t(), v) + qp.f
        r2 = qp.h - torch.matmul(qp.G, x)
        r3 = torch.min(qp.b-torch.matmul(qp.A,x), v)
        E = torch.norm(torch.cat([r1,r2,r3]))
        return E
    elif cfg.data_type == "svm":
        r1 = torch.matmul(qp.H, x) + torch.matmul(qp.A.t(), v) + qp.f
        r3 = torch.min(qp.b-torch.matmul(qp.A,x), v)
        print(r1.shape)
        print(r3.shape)
        E = torch.norm(torch.cat([r1,r3]))
        return E

def phi(a,b,alpha):
    """
    Surrogate function
    """
    yfb = a + b - torch.sqrt(a**2 + b**2)
    ypen = torch.max(torch.zeros(a.shape), a)*torch.max(torch.zeros(b.shape), b)
    y = alpha * yfb + (1-alpha)*ypen
    return y

def dphi(a,b,alpha):
    """
    derivative of surrogate function
    """
    a = a.squeeze()
    b = b.squeeze()
    q = a.shape[0]
    ztol = 100 * eps
    r = torch.sqrt(a**2 + b**2)
    S0 = (r <= ztol).float() 
    S1 = (a>0).float() * (b>0).float() 
    S2 = torch.ones(S0.shape) - S0 - S1 # 统计S0 S1 S2的比例，S0表示改变的种类不多
    g_s0 = alpha * torch.ones(a.shape) * (1-1/math.sqrt(2))

    g_s1 = alpha * (1-a/r) + (1-alpha) * b
    g_s2 = alpha * (1-a/r)
    
    S3 = (g_s1 < 1e-6).float() # 筛选出gamma趋于0的值，表示这些约束远离边缘
    g_s3 = 1e-3
    m_s3 = alpha
    
    m_s0 = alpha * torch.ones(b.shape) * (1-1/math.sqrt(2))
    m_s1 = alpha * (1-b/r) + (1-alpha) * a
    m_s2 = alpha * (1-b/r)

    #gamma = S0 * g_s0 + (S1-S3) * g_s1 + S2 * g_s2 + S3 * g_s3
    #mu = S0 * m_s0 + (S1-S3) * m_s1 + S2 * m_s2 + S3 * m_s3
    gamma = S0 * g_s0 + S1 * g_s1 + S2 * g_s2
    mu = S0 * m_s0 + S1 * m_s1 + S2 * m_s2
    return gamma, mu
    
def dphi_indexed(a,b,alpha,S_ch):
    """
    derivative of surrogate function with partial selection of constraints
    """
    a = a.squeeze()
    b = b.squeeze()
    q = a.shape[0]
    ztol = 100 * eps
    r = torch.sqrt(a**2 + b**2)
    S0 = (r <= ztol).float() 
    #print(torch.sum(S0))
    #print(S0.shape[0])
    #S1 = (a>-1e-4).float() * (b>-1e-4).float() 
    S1 = (a>0).float() * (b>0).float()
    S2 = torch.ones(S0.shape) - S0 - S1 # 统计S0 S1 S2的比例，S0表示改变的种类不多
    g_s0 = alpha * torch.ones(a.shape) * (1-1/math.sqrt(2))

    g_s1 = alpha * (1-a/r) + (1-alpha) * b
    g_s2 = alpha * (1-a/r)
    
    #S3 = torch.min(S1, (g_s1 < 1e-6).float()) # 筛选出gamma趋于0的值，表示这些约束远离边缘
    #S3 = (g_s1 < 1e-6).float()
    g_s3 = 1e-8
    m_s3 = alpha
    """
    S_prox = torch.zeros(S0.shape)
    if int(torch.sum(S1-S3)) > 4:
        idx_prox = torch.multinomial(S1-S3, int(torch.sum(S1-S3)/5), replacement = False)
        S_prox[idx_prox.long()] = 1

    S_vlt = torch.zeros(S0.shape)
    if int(torch.sum(S2)) > 1:
        #idx_vlt = torch.multinomial(S2, int(torch.sum(S2)/1.5), replacement = False)
        #S_vlt[idx_vlt.long()] = 1
        idx_vlt = torch.topk(torch.max(-(S2*a), -(S2*b)), int(torch.sum(S2)/1.5))
        S_vlt[idx_vlt[1]] = 1
    """
    m_s0 = alpha * torch.ones(b.shape) * (1-1/math.sqrt(2))
    m_s1 = alpha * (1-b/r) + (1-alpha) * a
    m_s2 = alpha * (1-b/r)

    gamma_pre = S0 * g_s0 + S1 * g_s1 + S2 * g_s2
    mu_pre = S0 * m_s0 + S1 * m_s1 + S2 * m_s2

    gamma_post = g_s3
    mu_post = m_s3

    #S_ch = torch.min(S_ch, S_prox)
    #print("------- attention here ------")
    #print(torch.sum(S_vlt), torch.sum(S2), torch.sum((S_ch > 0.5).float()))
    #S_vlt.detach()
    #S_ch = torch.max(S_ch, S_vlt)
    gamma = (1-S_ch) * gamma_post + S_ch * gamma_pre #没选中的用g_s3, 选中的用gamma_pre S_ch中为0的是没选中
    mu = (1-S_ch) * mu_post + S_ch * mu_pre
    #gamma = S0 * g_s0 + S_prox * g_s1 + S_vlt * g_s2 + (S1 - S_prox + S2 -S_vlt) * g_s3
    #mu = S0 * m_s0 + S_prox * m_s1 + S_vlt * m_s2 + (S1 - S_prox + S2 -S_vlt) * m_s3
    #mu = S0 * m_s0 + (S1-S3) * m_s1 + S2 * m_s2 + S3 * m_s3

    #S_indexed = (S_ch > 0.5).float()
    #S_prox = torch.zeros(S0.shape)
    """
    if int(torch.sum(S1-S3)) > 4:
        idx_prox = torch.multinomial(S1-S3, int(torch.sum(S1-S3)/5), replacement = False)
        S_prox[idx_prox.long()] = 1
        """
    #print("---prox:---")
    #print(torch.sum(torch.min(S1-S3, S_indexed)),torch.sum(S1-S3), torch.sum(S_indexed))
    
    #S_vlt = torch.min(S_indexed, S2)
    #print("---vlt:---")
    #print(torch.sum(S_vlt), torch.sum(S2),torch.sum(S_indexed))
    #gamma_indexed = S0 * g_s0 + S_prox * g_s1 + S_vlt * g_s2 + (S1 - S_prox + S2 -S_vlt) * g_s3
    #mu_indexed = S0 * m_s0 + S_prox * m_s1 + S_vlt * m_s2 + (S1 - S_prox + S2 -S_vlt) * m_s3
    return gamma, mu #, gamma_indexed, mu_indexed

def dphi_smw(a,b,alpha,S_ch,protype):
    """
    return the variation/reduction of selected constraints 
    """
    #原始的梯度都被构造为了 g=1e-3, m_s3=alpha
    #被选择的约束分别的梯度作为输出
    a = a.squeeze().cuda()
    b = b.squeeze().cuda()
    
    q = a.shape[0]
    ztol = 100 * eps
    r = torch.sqrt(a**2 + b**2)
    S0 = (r <= ztol).float()
    if torch.sum(S0) > 0:
        print("WARGNING: S0 has non-zero item.")
    S1 = (a>-1e-4).float() * (b>-1e-4).float() 
    S2 = torch.ones(S0.shape).cuda() - S0 - S1 # 统计S0 S1 S2的比例，S0表示改变的种类不多
    g_s0 = alpha * torch.ones(a.shape).cuda() * (1-1/math.sqrt(2)) + 1e-8
    g_s1 = alpha * (1-a/r) + (1-alpha) * b + 1e-8
    g_s2 = alpha * (1-a/r) + 1e-8
    
    S3 = torch.min(S1, (g_s1 < 1e-4).float()) # 筛选出gamma趋于0的值，表示这些约束远离边缘
    g_s3 = 1e-3 # + 1e-5
    m_s3 = alpha
    
    m_s0 = alpha * torch.ones(b.shape).cuda() * (1-1/math.sqrt(2))
    m_s1 = alpha * (1-b/r) + (1-alpha) * a
    m_s2 = alpha * (1-b/r)

    gamma_pre = S0 * g_s0 + S1 * g_s1 + S2 * g_s2
    mu_pre = S0 * m_s0 + S1 * m_s1 + S2 * m_s2
    S_ch_idx = np.where(S_ch > 0.5)[0] #选择的约束的index
    L_l = protype[:, S_ch_idx] * (alpha/1.01e-3 - mu_pre[S_ch_idx]/gamma_pre[S_ch_idx])
    L_r = protype[:, S_ch_idx]
    S_ch =S_ch.cuda()
    #gamma = S0 * g_s0 + S1 * g_s1 + S2 * g_s2
    #mu = S0 * m_s0 + S1 * m_s1 + S2 * m_s2
    gamma_post = g_s3
    mu_post = m_s3

    gamma = (1-S_ch) * gamma_post + S_ch * gamma_pre #没选中的用g_s3, 选中的用gamma_pre S_ch中为0的是没选中
    mu = (1-S_ch) * mu_post + S_ch * mu_pre
    #mu += 1e-2
    #return gamma.cpu(),mu.cpu(),L_l, L_r
    return gamma.cpu(), mu.cpu(), L_l, L_r

def dphi_reduced(a,b,alpha,S_ch):
    """
    return the variation/reduction of selected constraints 
    """
    #原始的梯度都被构造为了 g=1e-3, m_s3=alpha
    #被选择的约束分别的梯度作为输出
    a = a.squeeze()
    b = b.squeeze()
    q = a.shape[0]
    ztol = 100 * eps
    r = torch.sqrt(a**2 + b**2)
    S0 = (r <= ztol).float()
    if torch.sum(S0) > 0:
        print("WARGNING: S0 has non-zero item.")
    S1 = (a>-1e-4).float() * (b>-1e-4).float() 
    S2 = torch.ones(S0.shape) - S0 - S1 # 统计S0 S1 S2的比例，S0表示改变的种类不多
    
    g_s0 = alpha * torch.ones(a.shape) * (1-1/math.sqrt(2)) + 1e-5
    g_s1 = alpha * (1-a/r) + (1-alpha) * b #+ 1e-5
    g_s2 = alpha * (1-a/r) #+ 1e-5
    
    S3 = torch.min(S1, (g_s1 < 1e-4).float()) # 筛选出gamma趋于0的值，表示这些约束远离边缘
    g_s3 = 1e-3 #+ 1e-5
    m_s3 = alpha
    
    m_s0 = alpha * torch.ones(b.shape) * (1-1/math.sqrt(2))
    m_s1 = alpha * (1-b/r) + (1-alpha) * a
    m_s2 = alpha * (1-b/r)
    #print(S1)
    
    #print(torch.sum(S1))
    #print(torch.sum(S3))
    #print(torch.sum(S1-S3))
    S_prox = torch.zeros(S0.shape)
    L_prox_l = None
    L_prox_r = None
    prox_flag = False
    if int(torch.sum(S1-S3)) > 4:
        idx_prox = torch.multinomial(S1-S3, int(torch.sum(S1-S3)/5), replacement = False)
        S_prox[idx_prox.long()] = 1
        prox_flag = True
        L_prox_l = protype[:, idx_prox.long()] * (alpha/1.01e-3 -m_s1[idx_prox.long()]/g_s1[idx_prox.long()])
        L_prox_r = protype[:, idx_prox.long()]
    
    S_vlt = torch.zeros(S0.shape)
    L_vlt_l = None
    L_vlt_r = None
    vlt_flag = False
    if int(torch.sum(S2)) > 1:
        #idx_vlt = torch.multinomial(S2, int(torch.sum(S2)/1.5), replacement = False)
        #S_vlt[idx_vlt.long()] = 1
        #vlt_flag = True
        #L_vlt_l = protype[:, idx_vlt.long()] * (alpha/1.01e-3 - m_s2[idx_vlt.long()]/g_s2[idx_vlt.long()])
        #L_vlt_r = protype[:, idx_vlt.long()]
        #special_indices = np.where(alpha/1e-3 - m_s2[idx_vlt.long()]/g_s2[idx_vlt.long()] < 0)
        idx_vlt = torch.topk(torch.max(-(S2*a), -(S2*b)), int(torch.sum(S2)/1.5))
        S_vlt[idx_vlt[1]] = 1
        vlt_flag = True
        L_vlt_l = protype[:, idx_vlt[1]] * (alpha/1.01e-3 - m_s2[idx_vlt[1]]/g_s2[idx_vlt[1]])
        L_vlt_r = protype[:, idx_vlt[1]]
        
    
    if prox_flag and vlt_flag:
        L_l = torch.cat([L_prox_l,L_vlt_l], dim=1)
        L_r = torch.cat([L_prox_r,L_vlt_r], dim=1)
    elif vlt_flag:
        L_l = L_vlt_l
        L_r = L_vlt_r
    else:
        L_l = L_prox_l
        L_r = L_prox_r
    
    #gamma = S0 * g_s0 + (S1-S3) * g_s1 + S2 * g_s2 + S3 * g_s3
    #mu = S0 * m_s0 + (S1-S3) * m_s1 + S2 * m_s2 + S3 * m_s3

    #gamma = S0 * g_s0 + S1 * g_s1 + S2 * g_s2
    #mu = S0 * m_s0 + S1 * m_s1 + S2 * m_s2

    gamma = S0 * g_s0 + S_prox * g_s1 + S_vlt * g_s2 + (S1 - S_prox + S2 -S_vlt) * g_s3
    mu = S0 * m_s0 + S_prox * m_s1 + S_vlt * m_s2 + (S1 - S_prox + S2 -S_vlt) * m_s3
    print("reduced constraints: {}/{}".format(L_l.shape[1], S1.shape[0]))
    return gamma, mu, L_l, L_r

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[1,:]]  # get values from relevant entries of dense matrix #broadcast!!
    return torch.sparse.FloatTensor(i, v * dv, s.size())

class ConstraintHealNet(nn.Module):
    def __init__(self,kdim, logging=False):
        """
        feature_num_con: # of features in a constraint
        """
        super(ConstraintHealNet, self).__init__()
        #if cfg.double_attention:
        #    self.layer_norm_emb = torch.nn.LayerNorm([500, 1, self.feature_num_con], elementwise_affine=cfg.elementwise_affine)
        #self.embed_cd = torch.nn.Linear(301 * 2, 301, bias = True)
        #self.affine_matrix = Parameter(Tensor(64, 301, 301), requires_grad = True)
        #self.multihead_att_1 = CoCoAttention(embed_dim=64, num_heads=8, dropout=0.2)
        self.multihead_att_2 = CoCoAttention(embed_dim=301, num_heads=8, dropout=0.2)
        self.affine_bias = Parameter(torch.empty(1))
        self.affine_coef = Parameter(torch.empty(1))
        #self.norm_bias = Parameter(torch.empty(1))
        #self.coord_att = CoordAttention(embed_dim=2, num_heads=1, dropout=0.2)
        self.coord_embed = nn.Sequential(
                            nn.Linear(2, 64),
                            #nn.Tanh(),
                            nn.ReLU(),
                            nn.Linear(64,1))
        #self.fc = nn.Linear(self.feature_num_con, 1)
        self.reset_parameters()

    
    def reset_parameters(self):
        constant_(self.affine_bias, 0.)
        constant_(self.affine_coef, 1.)
        #init.xavier_normal(self.affine_matrix)


    def forward(self,x,node,y,v,d,special_indices,vlt_argmax,chnet_time=None):
        """
        x: f_x * 1 1000
        node1: e * f_c 500 * 1000
        x = [y,v]
        """
        #self.feature_num_x = x.shape[0]
        #self.padding = torch.nn.ZeroPad2d((0, self.feature_num_x - x.shape[0], 0,0))
        self.feature_num_con = x.shape[0]
        #self.layer_norm_flex = torch.nn.LayerNorm([node.shape[0], self.feature_num_con], elementwise_affine=cfg.elementwise_affine)
        #x_pad = self.padding(x).squeeze()
        x_pad = x.squeeze()
        if cfg.logging:
            #layer_norm_start_time = time.time()
            #
            #torch.cuda.synchronize()
            to_sparse_start_time = time.time()
            torch.cuda.synchronize()

        #key = self.embed_cd(torch.cat([node,d.expand(node.shape[0], -1)], dim = -1)).unsqueeze(1)
        #key1 = torch.matmul(node, torch.matmul(self.affine_matrix, d).t()).unsqueeze(1)
        #key2 = ((self.layer_norm_flex(node * x_pad) + self.norm_bias)* v).unsqueeze(1)
        #key2 = ((node * x_pad)*v).unsqueeze(1)
        key2 = (node * x_pad).unsqueeze(1)
        #key = (key2 - torch.mean(key2)) / torch.var(key2)
        key = key2
        print("before attention:")
        #print(torch.var(key2))
        print(np.percentile(torch.sum(key, dim = 1).detach().cpu().numpy(), [0,20,40,60,80,100]))
        print("d argmax before attention:",torch.sum(key2.squeeze(), dim = 1)[vlt_argmax])
        if cfg.logging:
            torch.cuda.synchronize()
            to_sparse_end_time = time.time()
            chnet_time[9] += to_sparse_end_time - to_sparse_start_time
            #torch.cuda.synchronize()
            #layer_norm_end_time = time.time()
            #chnet_time[6] += layer_norm_end_time - layer_norm_start_time
            att_time_start = time.time()
            torch.cuda.synchronize()
        
        emb, attn_weights = self.multihead_att_2(key, key, key)
        
        if cfg.double_attention:
            emb = self.layer_norm_emb(emb)
            emb, _, chnet_time = self.multihead_att_2(emb, emb, emb, chnet_time = chnet_time)
        if cfg.logging:
            torch.cuda.synchronize()
            att_time_end = time.time()
            chnet_time[4] += att_time_end - att_time_start

        """
        Attention后d在这里区别大吗？
        """
        #print("D after Attention:")
        #print(emb.squeeze())
        #print(emb.shape)
        #d = torch.mean(emb.squeeze(), 1)
        d = emb.squeeze()
        if cfg.logging:
            layer_norm_start_time = time.time()
            torch.cuda.synchronize()
        self.layer_norm_d_flex = torch.nn.LayerNorm([node.shape[0]], elementwise_affine=cfg.elementwise_affine)
        
        print("d after attention/before layer norm percentile:")
        print(np.percentile(d.detach().cpu().numpy(),[0,20,40,60,80,100]))
        print("v percentile:")
        print(np.percentile(v.detach().cpu().numpy(),[0,20,40,60,80,100]))
        print("d at argmax:", d[vlt_argmax])
        print("v at argmax:", v[vlt_argmax])
        #v_max_idx = torch.argmax(v)
        #print("IS V None???")
        #print(node[v_max_idx])
        #print(torch.sum((node[v_max_idx])))

        #d = torch.abs(d * v.squeeze())
        #d = torch.abs(d)
        print("MEAN ABS V:", 10 * torch.mean(torch.abs(v)))
        v_squeeze = torch.min(v.squeeze(), 10 * torch.mean(torch.abs(v).squeeze())) #TODO: torch.mean(torch.abs(v.squeeze()))
        v_squeeze = v.squeeze()
        
        if cfg._surrogate_func:
            d = torch.stack([d, v_squeeze], dim = -1).unsqueeze(1)
            d = self.coord_embed(d).squeeze()
        else:
            d = torch.abs(d + v_squeeze - torch.sqrt(d**2 + v_squeeze**2))
        
        #d = self.layer_norm_d_flex(d.squeeze()) * self.affine_coef
        #print(d.shape)
        #d, _ = self.coord_att(d, d, d)
        print("d after coord embed:")
        print(np.percentile(d.detach().cpu().numpy(),[0,20,40,60,80,100]))
        #d = torch.abs(d)
        
        print("d argmax surrogate:", d[vlt_argmax])

        #d = self.layer_norm_d_flex(d.squeeze()) * self.affine_coef # + self.affine_bias
        """
        attention for direction similarity
        """        
        #emb, attn_weights = self.multihead_att_1(key1, key1, key1)
        #d1 = self.layer_norm_d_flex(torch.mean(emb.squeeze(), 1))

        #print("History score:", d1)
        #d = d1 + d2
        d_scores = d
        if cfg.logging:
            torch.cuda.synchronize()
            layer_norm_end_time = time.time()
            chnet_time[6] += layer_norm_end_time - layer_norm_start_time
        print("d after layer norm percentile:")
        print(np.percentile(d.detach().cpu().numpy(),[0,20,40,60,80,100]))
        
        if cfg.logging:
            sigmoid_start_time = time.time()
            torch.cuda.synchronize()
        d_percentile = torch.sigmoid(d)
        #d_percentile = (d_percentile - torch.min(d_percentile).detach()) / (1 - torch.min(d_percentile).detach())
        #d_percentile = (d_percentile - torch.min(d_percentile).detach()) / (torch.max(d_percentile).detach() - torch.min(d_percentile).detach())
        #d_percentile = F.relu(torch.tanh(d_percentile))
        if cfg.logging:
            torch.cuda.synchronize()
            sigmoid_end_time = time.time()
            chnet_time[7] += sigmoid_end_time - sigmoid_start_time
        return d_scores, d_percentile,chnet_time

#def CHselect(CHNet,c_graph,cgraph_edge_list,x,num_constraints,num_iters):
    """
    Select Constraints for Updating
    Input: 
    CHNet: model
    c_graph: edge list with feature
    cgraph_edge_list:  LongTensor with indexes of edges
    x: iterate point
    num_constraints: number of constraints
    num_iters: number of iterations for propagation
    Output:
    bernoulli probability of each constraint
    """
def CHSelect(CHNet, c_feature, x_iterate, y, v, d, special_indices,vlt_argmax, chnet_time=None):
    cuda_start_time = time.time()
    d = d.squeeze()
    if cfg.cuda and not cfg.train:
        x_iterate = x_iterate.cuda()
        c_feature = c_feature.cuda()
        d = d.cuda()
        v = v.cuda()
    cuda_end_time = time.time()
    if chnet_time:
        chnet_time[5] += cuda_end_time - cuda_start_time
    d_scores, z, chnet_time = CHNet(x_iterate, c_feature, y, v, d, special_indices,vlt_argmax,chnet_time)
    return d_scores, z.squeeze(), chnet_time

def smw_inverse(A,U,V):
    """
    B=A+UV
    B'=A'-A'U(I+VA'U)'VA'
    """
    s = V.shape[0]
    return A-A@U@torch.inverse(torch.eye(s).cuda()+V@A@U)@V@A

def set_epoch(self, epoch):
    if self.epoch_decay is True:
        probs = decay_prob(epoch, k=cfg._epoch_decay_k)
        print('swith to epoch {}, prob. -> {}'.format(epoch, probs))
    return probs

def decay_prob(self, i, or_type=3, k=3000):
    if or_type == 1:  # Linear decay
        or_prob_begin, or_prob_end = 1., 0.
        or_decay_rate = (or_prob_begin - or_prob_end) / 10.
        ss_decay_rate = 0.1
        prob = or_prob_begin - (ss_decay_rate * i)
        if prob < or_prob_end:
            prob_i = or_prob_end
            print('[Linear] schedule sampling probability do not change {}'.format(prob_i))
        else:
            prob_i = prob
            print('[Linear] decay schedule sampling probability to {}'.format(prob_i))

    elif or_type == 2:  # Exponential decay
        prob_i = numpy.power(k, i)
        print('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

    elif or_type == 3:  # Inverse sigmoid decay
        prob_i = k / (k + numpy.exp((i / k)))
        # print('[Inverse] decay schedule sampling probability to {}'.format(prob_i))
    #self.probs = prob_i
    return prob_i


def pfb(qp,z, _lambda,v, sigma,inner_tol, alpha, niters, max_iters, inverse_time, c_feature,x_iterate_gt,CHNet, optimizer, scheduler):
    """
    inner loop
    """
    start_time_pfb = time.time()
    # parameters
    lnm = 5
    lsmax = 20
    mrec = torch.zeros([lnm,1])
    max_inner_iters = 100
    eta = 1e-8
    beta = 0.7
    
    zbar = torch.clone(z)
    lambar = torch.clone(_lambda)
    vbar = torch.clone(v)
    if cfg.data_type == "simulation":
        m, n = qp.G.shape
    elif cfg.data_type == "svm":
        n = qp.H.shape[1]
    q, _ = qp.A.shape
    special_indices = torch.LongTensor([0])
    special_idx_arr = torch.zeros([qp.A.shape[0]])
    #K_inv = torch.inverse(K)
    nR_list = []
    obj_list = []
    x_list = []
    v_list = []
    imit_loss_list = []
    bce_loss_list = []


    constraint_num_list = []
    dz = torch.zeros(c_feature.shape[1])
    for j in range(max_inner_iters):
        print("== Inner iter: {} ==".format(j))
        CHNet.train()

        y = qp.b - torch.matmul(qp.A, z)

        y_far = float(torch.sum(y > 1e-3)) / y.shape[0] * 100
        y_violated = float(torch.sum(y < -1e-5)) / y.shape[0] * 100
        y_prox = 100 - y_far - y_violated
        y_sat = float(torch.sum (y > -1e-5)) / y.shape[0] * 100

        print("Calculating Residual...")
        if cfg.data_type == "simulation":
            r1 = -(torch.matmul(qp.H, z)+torch.matmul(qp.G.t(),_lambda)\
                +torch.matmul(qp.A.t(), v) + qp.f)
            r2 = -(-torch.matmul(qp.G, z) + qp.h)
            r3 = -phi(y,v,alpha)
            R = torch.cat([r1,r2,r3], dim = 0)
        elif cfg.data_type == "svm":
            r1 = -(torch.matmul(qp.H, z)+torch.matmul(qp.A.t(), v) + qp.f)
            r3 = -phi(y,v,alpha)
            R = torch.cat([r1,r3], dim = 0)

        nR = torch.norm(R)
        r3norm = torch.norm(r3)
        all_vlt = torch.abs(r3)
        print("R percentile:{}".format(nR.item()))
        print(np.percentile(R.detach().numpy(),[0,10,30,50,70,90,100]))
        print("r3 percentile:", np.percentile(r3.detach().numpy(),[0,10,30,50,70,90,100]))
        nR_list.append(float(nR))
        vlt_argmax = torch.argmax(torch.abs(r3).squeeze())
        print("residual max:", r3[vlt_argmax])
        print("original y at argamx:", y[vlt_argmax])
        print("original v at argamx:", v[vlt_argmax])
        
        end_time = time.time()

        #update network
        if niters >= 1:
            optimizer.zero_grad()
            print("distance of d and d_gt")
            print(np.percentile(torch.abs(d-d_true).detach().numpy(), [0,10,30,50,70,90,100]))
            imitation_loss = torch.nn.MSELoss()(d, d_true)

            l1Loss = torch.mean(torch.abs(S_ch))
            nRLoss = nR
            bceloss = F.binary_cross_entropy_with_logits(d_scores, d_target)
            gt_loss = torch.nn.MSELoss()(d.squeeze(), d_opt) #g
            print("Inner iter: {} Imitation loss: {} l1 loss: {} Residual Norm: {} gt loss: {} BCELoss: {}".format(j, imitation_loss.item(), l1Loss.item(), nR, gt_loss.item(), bceloss.item()))
            imit_loss_list.append(imitation_loss.item())
            bce_loss_list.append(bceloss.item())
            if cfg._use_gt_loss:
                loss = imitation_loss +nRLoss  #+ l1Loss #+ nRLoss #g gt_loss 
            else:
                loss = imitation_loss + bceloss
            loss.backward()
            """
            for name,param in CHNet.named_parameters():
                print('层:',name,param.size())
                print('权值梯度',param.grad)
            """
            total_norm = torch.nn.utils.clip_grad_norm(CHNet.parameters(), 1e2)
            print("Back Propagating....")
            print("Gradient TOTAL NORM:", total_norm)
            optimizer.step()
            scheduler.step()

            r1.detach_()
            r3.detach_()
            z.detach_()
            y.detach_()
            v.detach_()
            dz.detach_()
            if cfg.data_type == "simulation":
                _lambda.detach_()
            R.detach_()
            R_ch.detach_()
                       
        #x_iterate = torch.cat([z,v], dim=0)
        x_iterate = z
        # CHNet for finding constraints
        vlt_argmax = torch.argmax(torch.abs(r3).squeeze())
        print("Selecting Constraints...")
        d_scores, S_ch, _ = CHSelect(CHNet, c_feature, x_iterate, y, v, dz,special_indices,vlt_argmax)
        d_target = (all_vlt > 1e-6).squeeze().float().detach()
        print("violated num in r3:",torch.sum(d_target))
        #S_ch = torch.zeros(S_ch.shape)
        #S_ch_idx = torch.multinomial(torch.ones(S_ch.shape), int(S_ch.shape[0]//2))
        #S_ch[S_ch_idx] = 1
        print("S_ch:")
        print(np.percentile(S_ch.detach().numpy(), [0,10,30,50,70,90,100]))

        if cfg._gumbel_softmax:
            S_ch = F.gumbel_softmax(torch.cat([S_ch, -S_ch], dim=1), hard=True)[:, 0]

        print("Selected Constraints:{}/{} Max SCH:{}".format(torch.sum(S_ch>0.5), S_ch.shape[0], torch.max(S_ch)))
        constraint_num_list.append(int(torch.sum(S_ch>0.5)))
        special_indices=torch.LongTensor(np.where(S_ch > 0.5))
        special_idx_arr_2 = (S_ch > 0.5).float()
        ch_vlt = all_vlt[special_indices]
        print("All Violation:", np.percentile(all_vlt.detach().numpy(),[0,10,30,50,70,90,100]))
        if torch.sum(special_idx_arr_2) > 0:
            print("Violation selected by S_ch:", np.percentile(ch_vlt.detach().numpy(),[0,10,30,50,70,90,100]))
        else:
            print("None is choosed by S_ch!")

        overlap_indices = torch.min(special_idx_arr, special_idx_arr_2)
        print("overlap num: {} idx1: {} idx2: {}".format(torch.sum(overlap_indices), torch.sum(special_idx_arr), torch.sum(special_idx_arr_2)))
        special_idx_arr = special_idx_arr_2
        
        gamma_true, mu_true = dphi(y,v,alpha)
        gamma, mu = dphi_indexed(y,v,alpha,S_ch)
        print("gamma distance:", torch.mean(torch.abs(gamma-gamma_true)))
        print("mu distance:", torch.mean(torch.abs(mu-mu_true)))
        #new iteration
        if cfg.data_type == "simulation":
            m, n = qp.G.shape
        elif cfg.data_type == "svm":
            n = qp.H.shape[1]
        # 原始K
        if cfg.data_type == "simulation":
            K = torch.cat([
                torch.cat([qp.H, qp.G.t(), qp.A.t()], dim = 1),
                torch.cat([-qp.G, torch.zeros(m, m+q)], dim = 1),
                torch.cat([-torch.matmul(torch.diag(gamma_true), qp.A), torch.zeros([q, m]), torch.diag(mu_true)], dim=1)
            ])       
            K = K + torch.eye(K.shape[0]) * 0.01
            start_time_inv = time.time()    
            K_inv_true = torch.inverse(K)
            end_time_inv = time.time()
            inverse_time += end_time_inv - start_time_inv
            d_true = torch.matmul(K_inv_true, R)
            d_true.detach_()
            [dz_true, dlam_true, dv_true] = torch.split(d_true,[n,m,q])

            # CHNet K
            K = torch.cat([
                torch.cat([qp.H, qp.G.t(), qp.A.t()], dim = 1),
                torch.cat([-qp.G, torch.zeros(m, m+q)], dim = 1),
                torch.cat([-torch.matmul(torch.diag(gamma), qp.A), torch.zeros([q, m]), torch.diag(mu)], dim=1)
            ])       
            K = K + torch.eye(K.shape[0]) * 0.01
            start_time_inv = time.time()    
            K_inv = torch.inverse(K)
        elif cfg.data_type == "svm":
            K = torch.cat([
                torch.cat([qp.H + 0.01 * torch.eye(qp.H.shape[0]), qp.A.t()], dim = 1),
                torch.cat([-torch.matmul(torch.diag(gamma_true), qp.A),  torch.diag(mu_true + 0.01 * gamma_true)], dim=1)
            ])
            """
            K = torch.cat([
                torch.cat([qp.H, qp.A[S_ch_idx].t()], dim = 1),
                torch.cat([-torch.matmul(torch.diag(gamma_true), qp.A)[S_ch_idx],  torch.diag(mu_true[S_ch_idx])], dim=1)
            ])
            """  
            #K = K + torch.eye(K.shape[0]) * 0.01
            start_time_inv = time.time()    
            K_inv_true = torch.inverse(K)
            end_time_inv = time.time()
            inverse_time += end_time_inv - start_time_inv
            #d_true = torch.matmul(K_inv_true, torch.cat([R[:n], R[n:][S_ch_idx]]))
            d_true = torch.matmul(K_inv_true, R)
            d_true.detach_()
            #[dz_true, dv_true] = torch.split(d_true,[n,q])
            #[dz_true, dv_true] = torch.split(d_true,[n,S_ch_idx.shape[0]])
            #dv_true_temp = torch.zeros([q,1])
            #dv_true_temp[S_ch_idx] = dv_true
            #d_true = torch.cat([dz_true, dv_true_temp])

            # CHNet K
            K = torch.cat([
                torch.cat([qp.H + 0.01 * torch.eye(qp.H.shape[0]),  qp.A.t()], dim = 1),
                torch.cat([-torch.matmul(torch.diag(gamma), qp.A), torch.diag(mu + 0.01 * gamma)], dim=1)
            ])
            #K = K + torch.eye(K.shape[0]) * 0.01
            start_time_inv = time.time()    
            K_inv = torch.inverse(K)
            S_ch = S_ch.unsqueeze(1)
            R_ch = torch.cat([r1,r3*S_ch], dim = 0)

        end_time_inv = time.time()
        inverse_time += end_time_inv - start_time_inv
        
        d = torch.matmul(K_inv, R_ch)
        if cfg.data_type == "simulation":
            [dz, dlam, dv] = torch.split(d,[n,m,q])
        elif cfg.data_type == "svm":
            if cfg._sample_with_decay:
                p = random.random()
                if p > probs:
                    [dz, dv] = torch.split(d,[n,q])
                else:
                    [dz, dv] = torch.split(d_true, [n,q])
            else:
                [dz, dv] = torch.split(d,[n,q])
        
        if cfg.data_type == "simulation":
            d_opt = x_iterate_gt-torch.cat([z,_lambda,v], dim=0).squeeze().detach()
        elif cfg.data_type == "svm":
            d_opt = x_iterate_gt-torch.cat([z,v], dim=0).squeeze().detach()
        if cfg._clamp_v:
            d_opt = torch.clamp(d_opt, -0.1, 0.1)

        print("Updating Variables...")
        t = 1
        z = z + t* dz
        v = v + t* dv
        #v = torch.min(v + t * dv, 5 * torch.mean(torch.abs(v)))
        #v = torch.min(v + t * dv, 1.1 * torch.max(v))
        if cfg.data_type == "simulation":
            _lambda = _lambda + t*dlam
        x_list.append(z.squeeze().detach().numpy())
        v_list.append(v.squeeze().detach().numpy())
        obj = z.t() @ qp.H @ z + z.t() @ qp.f
        obj_list.append(float(obj[0][0]))

        print("MEAN WITH GT:", torch.mean(abs(z.squeeze() - x_iterate_gt[:qp.H.shape[0]])) )
        niters = niters+1
    v = torch.max(torch.zeros(v.size()),v)

    return z, _lambda, v, niters, inverse_time, nR, nR_list, obj_list,constraint_num_list,x_list,v_list,imit_loss_list,bce_loss_list

def solve_qp(qp,x0,lambda0,v0,inverse_time,c_feature,x_iterate_gt,CHNet,optimizer,scheduler,cfg):
    # outer loop
    x = x0
    _lambda = lambda0
    v = v0
    E0 = nr(x, _lambda,v,qp) #检查是否达到拉格朗日鞍点
    sigma = 0.5
    
    # initialize
    dx = torch.ones(x.shape)
    dlambda = torch.ones(_lambda.shape)
    dv = torch.ones(v.shape)
    prox_iters = 0
    newton_iters = 0
    print(x_iterate_gt.shape)
    for j in range(max_prox_iters):
        E = nr(x,_lambda,v,qp)
        if (E <= tol + E0*rtol) or torch.norm(dx) + torch.norm(dlambda) + torch.norm(dv) <= dtol:
            print("Terminate with code:0")
            break
        elif newton_iters > max_newton_iters:
            print("Terminate with code:1")
            break
        xp, lambdap, vp, newton_iters, inverse_time, nR,nR_list,obj_list,constraint_num_list,x_list,v_list,imit_loss_list,bce_loss_list = pfb(qp, x, _lambda, v, sigma, inner_tol, alpha, \
                                                                newton_iters, max_newton_iters, inverse_time, \
                                                                c_feature,x_iterate_gt, CHNet, optimizer, scheduler)
        
        dx = xp - x
        dv = vp - v
        dlambda = lambdap - _lambda
        x = xp
        _lambda = lambdap
        v = vp
        
        prox_iters += 1
    return x, inverse_time, nR_list,obj_list,constraint_num_list,x_list,v_list,imit_loss_list,bce_loss_list

def pfb_test(qp, z, _lambda, v, sigma, inner_tol, alpha, niters, max_iters, inverse_time,chnet_time, c_feature,x_iterate_gt, CHNet,protype):
    """
    inner loop
    """
    start_time_pfb = time.time()
    # parameters
    lnm = 5
    lsmax = 20
    mrec = torch.zeros([lnm,1])
    max_inner_iters = 100
    eta = 1e-8
    beta = 0.7
    
    zbar = torch.clone(z)
    lambar = torch.clone(_lambda)
    vbar = torch.clone(v)
    if cfg.data_type == "simulation":
        m, n = qp.G.shape
    elif cfg.data_type == "svm":
        n = qp.H.shape[1]
        q = qp.A.shape[0]
    special_indices = torch.LongTensor([0])
    special_idx_arr = torch.zeros([qp.A.shape[0]])
    nR_list = []
    obj_list = []
    x_list = []
    v_list = []
    constraint_num_list = []
    #K_inv = torch.inverse(K)
    S_ch = torch.zeros([q])
    dz = torch.zeros(c_feature.shape[1])
    """
    if cfg.data_type == "simulation":
        K = torch.cat([
                torch.cat([qp.H, qp.G.t(), qp.A.t()], dim = 1),
                torch.cat([-qp.G, torch.zeros(m, m+q)], dim = 1),
                torch.cat([-qp.A, torch.zeros([q, m]), torch.eye(q,q)*alpha/1.01e-3], dim=1)
            ])
        #K_protype = K + torch.eye(K.shape[0]) * 0.01
        K_protype = (K + torch.eye(K.shape[0]) * 0.01).cuda()
        K_protype_inv = torch.inverse(K_protype)
    elif cfg.data_type == "svm":
        K = torch.cat([
                torch.cat([qp.H, qp.A.t()], dim = 1),
                torch.cat([-qp.A, torch.eye(q,q)*alpha/1.01e-3], dim=1)
            ])
        #K_protype = K + torch.eye(K.shape[0]) * 0.01
        K_protype = (K + torch.eye(K.shape[0]) * 0.01).cuda()
        K_protype_inv = torch.inverse(K_protype)
    """
    for j in range(max_inner_iters):
        print("=== Iter: {} ===".format(j))
        
        y = qp.b - torch.matmul(qp.A, z)
        print("****")
        print(z.requires_grad)
        print("****")
        #print("y:", y.squeeze())
        #print(qp.b.squeeze())
        if cfg.data_type == "simulation":
            r1 = -(torch.matmul(qp.H, z)+torch.matmul(qp.G.t(),_lambda)\
                +torch.matmul(qp.A.t(), v) + qp.f)
            r2 = -(-torch.matmul(qp.G, z) + qp.h)
            r3 = -phi(y,v,alpha)
            R = torch.cat([r1,r2,r3], dim = 0)
        elif cfg.data_type == "svm":
            r1 = -(torch.matmul(qp.H, z)+torch.matmul(qp.A.t(), v) + qp.f)
            r3 = -phi(y,v,alpha)
            R = torch.cat([r1,r3], dim = 0)

        nR = torch.norm(R)
        if cfg.output:
            nR_list.append(R.detach().numpy())

        # CHNet for finding constraints
        #x_iterate = torch.cat([z,v], dim=0)
        x_iterate = z
        chnet_start_time = time.time()
        if cfg.logging:
            torch.cuda.synchronize()

        c_feature_indices = torch.LongTensor(np.arange(qp.A.shape[0]))
        all_vlt = torch.abs(r3)
        vlt_argmax = torch.argmax(all_vlt)
        print("vlt shape:", all_vlt.shape)
        print("MAX Violated:", all_vlt[vlt_argmax])
        print("Y shape:", y.shape)
        print("Y at argmax:", y[vlt_argmax])
        print("V at argmax:", v[vlt_argmax])
        print("c feature shape:", c_feature.shape)
        print("x iterate shape:", x_iterate.shape)
        print("Az at argmax:", torch.matmul(qp.A,z)[vlt_argmax])
        print("cx at argmax:", torch.sum(c_feature * x_iterate.squeeze(), dim=1)[vlt_argmax])

        if cfg._duplicate_constraints and j % 5 != 0:
            c_feature_indices = duplicate_indices 
        #print("Duplicate Constraints:")
        #print(c_feature_indices.shape)
        
        if j > cfg._newton_iters_prelearn:
            with torch.no_grad():
                _, S_ch_renewed, chnet_time = CHSelect(CHNet, c_feature[c_feature_indices], x_iterate, y, v, dz, special_indices,vlt_argmax, chnet_time)#.cpu()
                S_ch[c_feature_indices] = S_ch_renewed.cpu()
        else:
            S_ch = torch.ones(S_ch.shape)
        print("S_ch:")
        print(np.percentile(S_ch.detach().numpy(), [0,10,30,50,70,90,100]))
        gamma_indexed, mu_indexed = dphi_indexed(y,v,alpha,S_ch)
        if cfg.logging:
            cpu_start_time = time.time()
            torch.cuda.synchronize()
        if cfg.cuda:
            S_ch = S_ch.cpu()
        if cfg.logging:
            torch.cuda.synchronize()
            cpu_end_time = time.time()
            chnet_time[8] += cpu_end_time - cpu_start_time
            torch.cuda.synchronize()
        chnet_end_time = time.time()
        chnet_time[0] += chnet_end_time - chnet_start_time
        #indices_selected = np.where(S_ch > 0.5)[0]
        #indices_selected = torch.topk(S_ch, 300)[1].long()
        indices_selected = torch.LongTensor(np.arange(qp.A.shape[0]))
        #print(len(indices_selected))
        #duplicate_indices = np.where(S_ch > 0.42)[0]
        if cfg._duplicate_constraints:
            duplicate_indices = torch.topk(S_ch, qp.A.shape[0] / 5)[1].long()

        cfg._index_num_bound = int(qp.A.shape[0] / 5)
        if j > cfg._newton_iters_prelearn and len(indices_selected) > cfg._index_num_bound: #g
            indices_selected = torch.topk(S_ch, cfg._index_num_bound)[1] #.detach().numpy()

        # hard argmax
        if cfg.output:
            constraint_num_list.append(torch.sum(indices_selected))
        #print("Selected Constraints:{}/{} Max SCH:{}".format(len(indices_selected), S_ch.shape[0], torch.max(S_ch)))
        S_ch = torch.zeros(S_ch.shape)
        #S_ch[indices_selected] = 1
        
        #special_indices=torch.LongTensor(np.where(S_ch > 0.5))
        #special_idx_arr_2 = (S_ch > 0.5).float()
        #ch_vlt = torch.max(-y, -v)[special_indices]
        #all_vlt = torch.max(-y, -v)
        #all_vlt = torch.abs(r3)
        #print(cfg._index_num_bound)
        #indices_selected = torch.topk(all_vlt.squeeze(), cfg._index_num_bound)[1].detach().long()
        #indices_selected = torch.multinomial(torch.ones(S_ch.shape), cfg._index_num_bound)
        ch_vlt = all_vlt[indices_selected]
        #print(np.sort(indices_selected))
        S_ch[indices_selected] = 1
        print("****")
        print(cfg._index_num_bound)
        print(indices_selected.shape)
        #print(torch.sum(indices_selected))
        print(ch_vlt.shape)
        print(torch.sum(all_vlt.squeeze()))
        print(torch.sum(ch_vlt.squeeze()))
        topk_indices = torch.topk(all_vlt.squeeze(), cfg._index_num_bound)[1].detach().long()
        print(torch.sum(all_vlt[topk_indices]))
        print("****")
        
        #print(indices_selected)
        #print(S_ch)
        print("Y violated num:", torch.sum(y < 0))
        print("Y percentile:", np.percentile(y.detach().numpy(),[0,10,30,50,70,90,100]))
        print("V violated num:", torch.sum(v < 0))
        print("V percentile:", np.percentile(v.detach().numpy(),[0,10,30,50,70,90,100]))
        print("YV violated num:", torch.sum(torch.abs(y * v) > 1e-5))
        print("YV percentile:", np.percentile(torch.abs(y*v).detach().numpy(),[0,10,30,50,70,90,100]))
        print("Y > 0, V > 0:", torch.sum((y>1e-5).float() * (v>1e-5).float()))

        print("All Violation:", np.percentile(all_vlt.detach().numpy(),[0,10,30,50,70,90,100]))
        if torch.sum(indices_selected) > 0:
            print("Violation selected by S_ch:", np.percentile(ch_vlt.detach().numpy(),[0,10,30,50,70,90,100]))
        else:
            print("None is choosed by S_ch!")
        
        print(torch.mean(abs(z.squeeze() - x_iterate_gt[:qp.H.shape[0]])))

        #S_ch = torch.zeros(S_ch.shape)
        #S_ch_idx = torch.multinomial(torch.ones(S_ch.shape), int(S_ch.shape[0]//1))
        #S_ch[S_ch_idx] = 1

        gamma_true, mu_true = dphi(y,v,alpha)
        #gamma, mu, gamma_indexed, mu_indexed = dphi_indexed(y,v,alpha,S_ch)
        
        #gamma,mu,L_l, L_r = dphi_smw(y,v,alpha,S_ch) # 输入
        dphi_start_time = time.time()
        gamma,mu, L_l, L_r = dphi_smw(y,v,alpha,S_ch,protype)
        dphi_end_time = time.time()
        chnet_time[1] += dphi_end_time - dphi_start_time
        #print("gamma indexed:", gamma_indexed)
        #print("gamma:", gamma)
        #print(np.percentile(y.detach().numpy(), [0,10,30,50,70,90]))
        #print(np.percentile(gamma.detach().numpy(),[0,10,30,50,70,90]))
                
        #gamma_acc_zero = float(torch.sum(gamma < 1.1 * 1e-3)) / gamma.shape[0] * 100

        #new iteration
        if cfg.data_type == "simulation":
            m, n = qp.G.shape
        elif cfg.data_type == "svm":
            n = qp.H.shape[1]
        q, _ = qp.A.shape

        start_time_inv = time.time()
        
        if cfg.smw_acc:
            K_inv = smw_inverse(K_protype_inv, L_l.cuda(), -L_r.cuda().t())
            K_inv = K_inv.cpu()
        else:
            #CHNet K
            if cfg.data_type == "simulation":
                K = torch.cat([
                    torch.cat([qp.H, qp.G.t(), qp.A.t()], dim = 1),
                    torch.cat([-qp.G, torch.zeros(m, m+q)], dim = 1),
                    torch.cat([-qp.A, torch.zeros([q, m]), torch.diag(mu/gamma)], dim=1)
                ])
                #print(K)
            elif cfg.data_type == "svm":
                K = torch.cat([
                    torch.cat([qp.H + 0.01 * torch.eye(qp.H.shape[0]), qp.A.t()], dim = 1),
                    torch.cat([-torch.matmul(torch.diag(gamma_true), qp.A),  torch.diag(mu_true + 0.01 * gamma_true)], dim=1)
                ])
                """
                K = torch.cat([
                    torch.cat([qp.H, qp.A[S_ch_idx].t()], dim = 1),
                    torch.cat([-torch.matmul(torch.diag(gamma_true), qp.A)[S_ch_idx],  torch.diag(mu_true[S_ch_idx])], dim=1)
                ])
                """  
                start_time_inv = time.time()    
                K_inv_true = torch.inverse(K)
                R = torch.cat([r1, r3], dim=0)
                d_true = torch.matmul(K_inv_true, R)
                #print("IN SVM K")
                K = torch.cat([
                    torch.cat([qp.H + 0.01 * torch.eye(qp.H.shape[0]), qp.A.t()], dim = 1),
                    torch.cat([-torch.matmul(torch.diag(gamma_indexed), qp.A),  torch.diag(mu_indexed + 0.01 * gamma_indexed)], dim=1)
                ])
                K_inv_indexed = torch.inverse(K).cpu()
                K = torch.cat([
                    torch.cat([qp.H + 0.01 * torch.eye(qp.H.shape[0]), qp.A.t()], dim = 1),
                    torch.cat([-torch.matmul(torch.diag(gamma), qp.A),  torch.diag(mu + 0.01 * gamma)], dim=1)
                ])
                
                #print("K:", K)
            #K = (K + torch.eye(K.shape[0]) * 1e-2).cuda()
            
            K_inv = torch.inverse(K).cpu()
        end_time_inv = time.time()
        inverse_time += end_time_inv - start_time_inv

        if cfg.data_type == "simulation":
            R = torch.cat([r1,r2,torch.matmul(torch.diag(1/gamma), r3)], dim = 0)
            d = torch.matmul(K_inv, R)
            [dz, dlam, dv] = torch.split(d,[n,m,q])
        elif cfg.data_type == "svm":
            #R = torch.cat([r1, torch.matmul(torch.diag(S_ch/gamma), r3)], dim=0)
            R = torch.cat([r1, torch.matmul(torch.diag(S_ch), r3)], dim=0)
            d = torch.matmul(K_inv, R)
            d_indexed = torch.matmul(K_inv_indexed, R)
            imitation_loss = torch.nn.MSELoss()(d, d_true)
            imitation_loss_indexed = torch.nn.MSELoss()(d_indexed, d_true)
            similarity_indexed = torch.nn.MSELoss()(d_indexed, d)
            print("Imitation loss with rounded:", imitation_loss)
            print("Imitation loss with unrounded:", imitation_loss_indexed)
            print("Similarity of rouned and unrounded:", similarity_indexed)
            #print("NORM D:", torch.norm(d))
            #print("NORM K_inv:", torch.norm(K_inv))
            #[dz, dv] = torch.split(d,[n,q])
            [dz, dv] = torch.split(d_indexed, [n,q])
        t = 1
        print("NORM Dz:",torch.norm(dz))
        print("NORM Dv:", torch.norm(dv))
        print("DV percentile:", np.percentile(dv.detach().numpy(), [0,10,30,50,70,90,100]))
        dv_argmax = torch.argmax(dv)
        print("DV MAX:", dv[dv_argmax])
        print("V at DV MAX:",v[dv_argmax])
        print("Selected R:", torch.norm(R))
        print("All R:", torch.norm(torch.cat([r1, r3], dim=0)))
        print("Most Violated:")
        print("R Max:", torch.max(torch.abs(r3)))
        argmax = np.argmax(torch.abs(r3).squeeze().numpy())
        print("Y at argamx:", y[argmax])
        print("V at argmax:", v[argmax])
        print("Phi at argmax:", phi(y[argmax], v[argmax], alpha))
        print("Y MIN:",np.min(y.squeeze().numpy()))
        print("Y ARGMIN:",np.argmin(y.squeeze().numpy()))
        argmax = np.argmin(y.squeeze().numpy())
        print("all vlt:", all_vlt[argmax])
        print("v:", v[argmax])
        print("Adz:", torch.matmul(qp.A[argmax], dz))
        residual = torch.norm(torch.matmul(K.cpu(), d) - R)
        print("mm residual:", residual)
        z = z + t* dz
        v = v + t *dv
        #v = torch.min(v + t * dv, 5 * torch.mean(torch.abs(v)))
        #v = torch.min(v + t * dv, 1.1 * torch.max(v))
        if cfg.data_type == "simulation":
            _lambda = _lambda + t*dlam
        if cfg.output:
            x_list.append(z.squeeze().detach().numpy())
            v_list.append(v.squeeze().detach().numpy())
            obj = z.t() @ qp.H @ z + z.t() @ qp.f
            obj_list.append(float(obj[0][0]))
        """
        if torch.mean(abs(z.squeeze() - x_iterate_gt[:qp.H.shape[0]])) < 0.01:
            break
        """
        niters = niters+1
    #print("End at Iter:", j)
    v = torch.max(torch.zeros(v.size()),v)

    return z, _lambda, v, niters, inverse_time, nR, nR_list, obj_list,constraint_num_list,x_list,v_list,chnet_time

def test_solver(qp,x0,lambda0,v0,inverse_time,chnet_time,c_feature,x_iterate_gt,CHNet,cfg):
    x = x0
    _lambda = lambda0
    v = v0

    CHNet.eval()
    dx = torch.ones(x.shape)
    dlambda = torch.ones(_lambda.shape)
    dv = torch.ones(v.shape)
    newton_iters = 0

    if cfg.data_type == "svm":
        q = qp.A.shape[0]
        m = qp.H.shape[0]
        protype = torch.cat([
            torch.zeros([m, q]),
            torch.eye(q)], dim = 0).cuda()
    elif cfg.data_type == "simulation":
        protype = torch.cat([
            torch.zeros([m*n+m, m*n]),
            torch.eye(m*n)], dim = 0).cuda()

    xp, lambdap, vp, newton_iters, inverse_time, nR, nR_list, obj_list,constraint_num_list,x_list,v_list,chnet_time = pfb_test(qp, x, _lambda, v, sigma, inner_tol, alpha, \
                                                                newton_iters, max_newton_iters, inverse_time, chnet_time, c_feature,x_iterate_gt, CHNet,protype)
    x = xp
    _lambda = lambdap
    v = vp
    return x, inverse_time,  nR_list,obj_list,constraint_num_list,x_list,v_list,chnet_time


def graph_generate(A,b):
    """
    A dim: q * n
    b: q * 1
    return
    c_graph: list 2 * num_edge * num_feature
    """
    #TODO: b is not used here
    q, n = A.shape
    sample_n = 10
    def corr(A):
        q, _ = A.shape
        ret = (torch.ones([q,q]) - torch.diag(torch.ones(q))) * (A @ A.t())
        return ret
    corr_A = torch.sigmoid(corr(A))
    graph_c = torch.multinomial(corr_A, num_samples=sample_n, replacement=False)
    edge_tx = torch.LongTensor(np.arange(0,q).repeat(sample_n)) # q
    edge_rx = torch.LongTensor(graph_c.reshape(-1))
    c_graph_edgelist = torch.stack([edge_tx, edge_rx], dim=0)
    c_graph = [A[c_graph_edgelist[0]], A[c_graph_edgelist[1]]]
    return c_graph, c_graph_edgelist
    
# redirect output to file

class Logger(object):
    def __init__(self, start_time):
        filename="logs/"+ start_time +".txt"
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()

if __name__ == '__main__':
    inverse_time = 0
    chnet_time = [0,0,0,0,0,0,0,0,0,0]
    #Hs = np.load("instances/qp_H.npy")
    #fs = np.load("instances/qp_f.npy")
    prefix = "/home/chenzhijie/"
    init_dir = prefix + "research/newtonacc/logs/svm/init/"
    constraints = []
    file_name_list = []
    x_iterate_gt = []
    x_inits = []
    #v_inits = []
    if cfg.data_type == "svm":
        data_dir = prefix + "research/newtonacc/data/svm/split"
        """
        files = os.listdir(data_dir)
        for filename in files:
            print(filename)
            constraints.append(np.load(data_dir + "/" + filename))
            file_name_list.append(filename)
            x_iterate_gt.append(torch.FloatTensor(np.load(prefix+"research/newtonacc/logs/svm/split/"+filename+"_z_opt.npy")))
        """
        for dataset_num in range(1,10):
            for batch_num in range(5):
                filename = "w{}a_{}.npy".format(dataset_num, batch_num)
                constraints.append(np.load(data_dir + "/" + filename))
                file_name_list.append(filename)
                x_iterate_gt.append(torch.FloatTensor(np.load(prefix+"research/newtonacc/logs/svm/split/"+filename+"_z_opt.npy")))
                x_inits.append(torch.FloatTensor(np.load(init_dir + filename + "_z_init.npy")))
                #v_inits.append(torch.FloatTensor(np.load(init_dir + filename + "_v_init.npy")))
        
    elif cfg.data_type == "simulation":
        Hs = np.load("instances/qp_H.npy")
        fs = np.load("instances/qp_f.npy")
        x_iterate_gt = torch.FloatTensor(np.load(prefix+"research/newtonacc/logs/svm/z_opt.npy"))

    obj_list = []
    nR_list = []
    #output_dim = qp.A.shape[1] + qp.A.shape[0] # do not consider b yet

    # train
    #feature_num_x = qp.G.shape[1] + qp.A.shape[0]
    #feature_num_con = qp.A.shape[1]
    
    # CHNet
    CHNet = ConstraintHealNet(kdim=cfg.kdim)
    CHNet.load_state_dict(torch.load(prefix+"research/newtonacc/model/2021-03-28-00:24:37/CHNet_40.pt"))

    optimizer = optim.SGD(CHNet.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY)
                                               #last_epoch=cfg.TRAIN.START_EPOCH - 1)
    
    start_time = time.time()
    timeArray = time.localtime(start_time)
    start_time_str = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)
    sys.stdout = Logger(start_time_str)
    if cfg.train:
        os.mkdir(prefix + "research/newtonacc/model/"+start_time_str)
        os.mkdir(prefix + "research/newtonacc/train/"+start_time_str)
    os.mkdir(prefix + "research/newtonacc/results/svm/"+start_time_str)
    shutil.copyfile('fbstab_nn_0202.py', 'logs/fbstab_nn_0202_'+start_time_str+'.py')
    nR_list_arr = []
    obj_list_arr = []
    x_list_arr = []
    v_list_arr = []
    constraint_num_list_arr = []
    imit_loss_list_arr = []
    bce_loss_list_arr = []

    
    if cfg.train:
        for epoch in range(100):
            for i in range(25):
                print("Start Training...Epoch:{} w{}a_{}".format(epoch, i//5, i%5+1))
                #H = torch.randn([m*n, m*n])
                #qp.H = torch.matmul(H.t(), H) + 0.01 * torch.eye(m*n)
                #qp.f = torch.randn([m*n, 1])
                qp = qp_struct()
                if cfg.data_type == "simulation":
                    qp.H = torch.FloatTensor(Hs[i])
                    qp.f = torch.FloatTensor(fs[i])
                    qp.G = torch.FloatTensor(np.eye(m).repeat(n, axis=1))
                    qp.h = torch.ones([m,1])
                    qp.A = -torch.eye(m*n)
                    #qp.A = -torch.FloatTensor(random(n,m*n,density=0.5).A)
                    qp.b = torch.zeros([m*n, 1])
                elif cfg.data_type == "svm":
                    qp.H = torch.eye(constraints[i].shape[1])   
                    qp.H[-1,-1] = 0
                    qp.f = torch.zeros([constraints[i].shape[1],1])
                    qp.G = None
                    qp.h = None
                    qp.A = -torch.FloatTensor(constraints[i])
                    qp.b = -torch.ones([constraints[i].shape[0],1])
                
                #x0 = torch.rand([qp.H.shape[0],1])

                if cfg.data_type == "simulation":    
                    lambda0 = torch.rand([qp.G.shape[0],1])
                elif cfg.data_type == "svm":
                    lambda0 = torch.rand([1,1])
                #v0 = torch.rand([qp.A.shape[0],1])
                
                [x0, v0] = torch.split(x_inits[i], [constraints[i].shape[1], qp.A.shape[0]])
                x0 = x0.unsqueeze(1)
                v0 = v0.unsqueeze(1)
                #print("===nR:===")
                #print(nr(x0,lambda0,v0,qp))
                #sys.exit()

                #c_feature= torch.cat([qp.A, torch.eye(qp.A.shape[0])], dim=1)
                c_feature = qp.A
                x, _ , nR_list,obj_list_from_qp,constraint_num_list,x_list,v_list,imit_loss_list,bce_loss_list = solve_qp(qp, x0, lambda0, v0, inverse_time,c_feature,x_iterate_gt[i],CHNet,optimizer,scheduler,cfg)
                
                #nR_list_arr.append(nR_list)
                #obj_list_arr.append(obj_list_from_qp)
                #constraint_num_list_arr.append(constraint_num_list)
                #x, inverse_time, nR = test_solver(qp,x0,lambda0,v0,inverse_time,CHNet,cfg)
                #x_list_arr.append(x_list)
                obj = x.t() @ qp.H @ x + x.t() @ qp.f
                obj_list.append(float(obj[0][0]))
                imit_loss_list_arr.append(imit_loss_list)
                bce_loss_list_arr.append(bce_loss_list)
            if (epoch+1)%1==0:
                torch.save(CHNet.state_dict(), prefix + "research/newtonacc/model/"+start_time_str+"/CHNet_"+str(epoch+1)+".pt")
                np.save(prefix+"research/newtonacc/train/"+start_time_str+"/imit_loss.npy", np.array(imit_loss_list_arr))
                np.save(prefix+"research/newtonacc/train/"+start_time_str+"/bce_loss.npy", np.array(bce_loss_list_arr))
                #sys.exit()
                #print(qp.G @ x - qp.h)
                #print(qp.A @ x + qp.b)
                #nR_list.append(nR)
        
    #torch.save(CHNet.state_dict(), "/home/chenzhijie/research/newtonacc/model/CHNet.pt")
    #CHNet.load_state_dict(torch.load("/home/chenzhijie/research/newtonacc/model/2021-01-29-08:08:57/CHNet_50.pt"))
    #
    

    if cfg.cuda and cfg.train:
        CHNet.load_state_dict(torch.load(prefix+'research/newtonacc/model/'+start_time_str+'/CHNet_8.pt', map_location=lambda storage, loc: storage.cuda(1)))
        CHNet.cuda()
    elif cfg.train:
        CHNet.load_state_dict(torch.load(prefix+"research/newtonacc/model/"+start_time_str+"/CHNet_10.pt"))
    elif cfg.cuda:
        CHNet.load_state_dict(torch.load(prefix+'research/newtonacc/model/2021-03-28-00:24:37/CHNet_40.pt', map_location=lambda storage, loc: storage.cuda(0)))
        CHNet.cuda()
    else:
        CHNet.load_state_dict(torch.load(prefix+"research/newtonacc/model/2021-02-21-06:40:51/CHNet_1.pt"))

    cfg.train = False
    
    if cfg.cuda:
        CHNet.cuda()

    CHNet.eval()
    #torch.no_grad()
    start_time = time.time()
    for i in range(len(file_name_list)):
        print("Start Testing...{}".format(file_name_list[i]))
        qp = qp_struct()
        if cfg.data_type == "simulation":
            qp.H = torch.FloatTensor(Hs[i])
            qp.f = torch.FloatTensor(fs[i])
            qp.G = torch.FloatTensor(np.eye(m).repeat(n, axis=1))
            qp.h = torch.ones([m,1])
            qp.A = -torch.eye(m*n)
            #qp.A = -torch.FloatTensor(random(n,m*n,density=0.5).A)
            qp.b = torch.zeros([m*n, 1])
        elif cfg.data_type == "svm":
            qp.H = torch.eye(constraints[i].shape[1])   
            qp.H[-1,-1] = 0
            qp.f = torch.zeros([constraints[i].shape[1],1])
            qp.G = None
            qp.h = None
            qp.A = -torch.FloatTensor(constraints[i])
            qp.b = -torch.ones([constraints[i].shape[0],1])
        #x0 = torch.rand([qp.H.shape[0],1])
        #x0 = x_inits[i].unsqueeze(-1)
        if cfg.data_type == "simulation":    
            lambda0 = torch.rand([qp.G.shape[0],1])
        elif cfg.data_type == "svm":
            lambda0 = torch.rand([1,1])
        #v0 = torch.rand([qp.A.shape[0],1])
        #v0 = v_inits[i].unsqueeze(-1)
        #[x0, v0] = torch.split(x_iterate_gt[i] * (torch.rand(x_iterate_gt[i].shape)+0.5), [qp.H.shape[0], qp.A.shape[0]])
        #[x0, v0] = torch.split(x_iterate_gt[i], [qp.H.shape[0], qp.A.shape[0]])
        
        [x0, v0] = torch.split(x_inits[i], [constraints[i].shape[1], qp.A.shape[0]])
        x0 = x0.unsqueeze(1)
        v0 = v0.unsqueeze(1)
        #c_feature= torch.cat([qp.A, torch.eye(qp.A.shape[0])], dim=1)
        c_feature = qp.A
        #x, inverse_time, nR = solve_qp(qp, x0, lambda0, v0, inverse_time,CHNet, SageNet, optimizer, scheduler,cfg)
        x, inverse_time, nR_list,obj_list_from_qp,constraint_num_list,x_list,v_list,chnet_time = test_solver(qp,x0,lambda0,v0,inverse_time,chnet_time,c_feature,x_iterate_gt[i],CHNet,cfg)
        nR_list_arr.append(nR_list)
        obj_list_arr.append(obj_list_from_qp)
        constraint_num_list_arr.append(constraint_num_list)
        x_list_arr.append(x_list)
        v_list_arr.append(v_list)
        obj = x.t() @ qp.H @ x + x.t() @ qp.f
        obj_list.append(float(obj[0][0]))

        np.save(prefix+"research/newtonacc/results/svm/"+start_time_str+"/"+file_name_list[i]+"_nR.npy",np.array(nR_list_arr[-1]))
        np.save(prefix+"research/newtonacc/results/svm/"+start_time_str+"/"+file_name_list[i]+"_obj.npy",np.array(obj_list_arr[-1]))
        np.save(prefix+"research/newtonacc/results/svm/"+start_time_str+"/"+file_name_list[i]+"_constraint.npy",np.array(constraint_num_list_arr[-1]))
        np.save(prefix+"research/newtonacc/results/svm/"+start_time_str+"/"+file_name_list[i]+"_x_iterate.npy",np.array(x_list_arr[-1]))
        np.save(prefix+"research/newtonacc/results/svm/"+start_time_str+"/"+file_name_list[i]+"_v_iterate.npy",np.array(v_list_arr[-1]))

    end_time = time.time()
    print("Total Time:{}".format(end_time-start_time))
    print("Total Inverse Time:{}".format(inverse_time))
    print("Total CHNet Time:{} dphi:{} Softmax Time:{} Out Proj:{} Attention Time:{} CUDA TIME:{} Layer Norm:{} Sigmoid:{} CPU TIME:{} SPARSE TIME:{}".format(\
            chnet_time[0],chnet_time[1],chnet_time[2],chnet_time[3],chnet_time[4],chnet_time[5],chnet_time[6],chnet_time[7],chnet_time[8],chnet_time[9]))
    #print(obj_list[:50])
    #print(obj_list[50:])
    """
    with open("/home/chenzhijie/research/newtonacc/logs/log_"+start_time_str+".txt", "w") as log_file:
        log_file.write("Total Time:{}\n".format(end_time-start_time))
        log_file.write("Total Inverse Time:{}\n".format(inverse_time))
        log_file.write("Total CHNet Time:{}".format(chnet_time[0]))
    """
    #print(nR_list)