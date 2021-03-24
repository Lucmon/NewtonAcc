import time
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model
        num_heads: parallel attention layers, or heads

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,vdim=None,_use_linformer=False,weight_dropout=True):
        super(MultiheadAttention, self).__init__()
        #self.embed_dim = embed_dim
        self.kdim = kdim
        self.head_dim = 64

        #self.num_heads = num_heads
        self.dropout = dropout
        self._use_linformer = _use_linformer
        self.weight_dropout = weight_dropout
        #self.head_dim = embed_dim // num_heads
        #assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        #self.head_dim = kdim // num_heads
        #assert self.head_dim * num_heads == self.kdim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        #self.in_proj_weight = Parameter(torch.empty(3 * kdim, embed_dim))
        self.in_proj_weight = Parameter(torch.empty(3 * kdim, self.head_dim))
        #self.kv_proj_weight = Parameter(torch.empty(32 ,500))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * kdim))
        else:
            self.register_parameter('in_proj_bias', None)
        #self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(kdim, self.head_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, kdim))
            self.bias_v = Parameter(torch.empty(1, 1, kdim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[:self.kdim, :])
        xavier_uniform_(self.in_proj_weight[self.kdim:(self.kdim * 2), :])
        xavier_uniform_(self.in_proj_weight[(self.kdim * 2):, :])

        xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None,chnet_time=None,logging=False):
        
        """
        Inputs of forward function
            query: [target length, batch size, embed dim]  ## QUERY is a sparse matrix
            key: [sequence length, batch size, embed dim]
            value: [sequence length, batch size, embed dim]
            key_padding_mask: if True, mask padding based on batch size
            incremental_state: if provided, previous time steps are cashed
            need_weights: output attn_output_weights
            static_kv: key and value are static

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        self.logging = logging
        tgt_len, bsz, embed_dim = query.size()
        #tgt_len, embed_dim = query.size()
        #bsz = 1
        #assert embed_dim == self.embed_dim
        #assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        self.num_heads = embed_dim // self.head_dim
        # padding
        if self.num_heads * self.head_dim < embed_dim:
            self.num_heads += 1
            self.padding = torch.nn.ZeroPad2d((0, self.num_heads * self.head_dim - embed_dim, 0,0))
            query = self.padding(query)
            key = self.padding(key)
            value = self.padding(value)
            embed_dim = self.num_heads * self.head_dim
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None
        
        if self.logging:
            proj_start_time = time.time()
            torch.cuda.synchronize()
        
        query = query.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        key = key.contiguous().view(-1, bsz * self.num_heads, self.head_dim)
        value = value.contiguous().view(-1, bsz * self.num_heads, self.head_dim)

        if qkv_same:
            # self-attention
            q, k, v = self._in_proj_qkv(query,key)
        elif kv_same:
            # encoder-decoder attention
            q = self._in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self._in_proj_kv(key)
        else:
            q = self._in_proj_q(query)
            k = self._in_proj_k(key)
            v = self._in_proj_v(value)
        #q, k, v = self._sparse_proj(query, key) # a sparse implementation
        q *= self.scaling

        if self.logging:
            torch.cuda.synchronize()
            proj_end_time = time.time()
            chnet_time[1] += proj_end_time - proj_start_time

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz * self.num_heads, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz * self.num_heads, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        #q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q = q.transpose(0, 1)
        if k is not None:
            #k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.transpose(0, 1)
        if v is not None:
            #v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        if self.logging:
            attetion_start_time = time.time()
            torch.cuda.synchronize()
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        #attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

        attn_output = torch.bmm(attn_output_weights, v)
        if self.logging:
            torch.cuda.synchronize()
            attention_end_time = time.time()
            chnet_time[2] += attention_end_time - attetion_start_time

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.kdim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads, self.kdim)

        if self.logging:
            out_proj_start_time = time.time()
            torch.cuda.synchronize()
        attn_output = self.out_proj(attn_output)
        #attn_output = torch.mean(attn_output, dim = 2)
        attn_output = attn_output.contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)
        if self.logging:
            torch.cuda.synchronize()
            out_proj_end_time = time.time()
            chnet_time[3] += out_proj_end_time - out_proj_start_time

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights, chnet_time


    def _in_proj_qkv(self, query, key):
        if self._use_linformer:
            k, v = self._in_proj_kv(key)
            return self._in_proj(query,end=self.kdim), k, v
        else:
            #print(self._in_proj(query).shape)
            return self._in_proj(query).chunk(3, dim=-1)

    def _in_proj_kv(self, key):
        #return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
        """
        self.kv_proj_weight: k * n
        key: n * 1 * d
        """
        key = torch.matmul(self.kv_proj_weight,key.permute([2,0,1])).permute([1,2,0])
        return self._in_proj(key, start=self.kdim).chunk(2, dim=-1)

    def _in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def _in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def _in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if self.weight_dropout:
            weight = F.dropout(weight, p=self.dropout)
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def _sparse_proj(self, query, key):
        """
        sparse implementation
        q, k, v: n* n
        
        """
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        kv_proj_weight = self.kv_proj_weight

        q_ = torch.sparse.mm(query, weight[:self.kdim,:].t()) + bias[:self.kdim]

        k_reduced = torch.sparse.mm(key.t(), kv_proj_weight.t()).t()
        #print(key.t().shape)
        #print(kv_proj_weight.t().shape)
        #print(k_reduced.shape)
        #print(weight[self.kdim:,:].t().shape)
        k_, v_ = (torch.matmul(k_reduced, weight[self.kdim:,:].t()) + bias[self.kdim:]).chunk(2, dim=-1)
        return q_.unsqueeze(1), k_.unsqueeze(1), v_.unsqueeze(1)