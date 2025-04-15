# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from jointformer.model.pos_embed import get_2d_sincos_pos_embed, get_abs_pos
from einops import rearrange
import torch.utils.checkpoint as cp

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 topk_num=None, topk_range='every_frame', topk_block=False, topk_clstoken=True,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.topk_num, self.topk_range, self.topk_block, self.topk_clstoken = topk_num, topk_range, topk_block, topk_clstoken

    def forward_feature(self, x, hw, T, backbone_update):
        """

        @param x: cls_token & query & memory [bs, 1 + hw + t*hw, 784]
        @param backbone_update: if True, cls_token do cross-attention in mix-attn block
        @return: cls_token & query & memory [bs, 1 + hw + t*hw, 784]
        """
        B, N, C = x.shape
        assert N == 1 + hw + T * hw
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
        q, k, v = qkv[0], qkv[1], qkv[2]   # [bs, nheads, N, C//nheads]
        clstoken_q, query_q, memory_q = torch.split(q, [1, hw, T * hw], dim=2)   # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]
        clstoken_k, query_k, memory_k = torch.split(k, [1, hw, T * hw], dim=2)   # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]
        clstoken_v, query_v, memory_v = torch.split(v, [1, hw, T * hw], dim=2)   # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]

        # 1) memory: self-attention
        memory_q_frame = rearrange(memory_q, 'b h (t n) c -> (b t) h n c', b=B, t=T, n=hw)  # [bs*t, nheads, hw, C//nheads]
        memory_k_frame = rearrange(memory_k, 'b h (t n) c -> (b t) h n c', b=B, t=T, n=hw)  # [bs*t, nheads, hw, C//nheads]
        memory_v_frame = rearrange(memory_v, 'b h (t n) c -> (b t) h n c', b=B, t=T, n=hw)  # [bs*t, nheads, hw, C//nheads]
        memory_attn = (memory_q_frame @ memory_k_frame.transpose(-2, -1)) * self.scale  # [bs*t, nheads, hw, hw]
        memory_attn = memory_attn.softmax(dim=-1)
        memory_attn = self.attn_drop(memory_attn)

        memory = (memory_attn @ memory_v_frame).transpose(1, 2).reshape(B * T, hw, C) # [bs*t, nheads, hw, C//nheads] -> [bs*t, hw, nheads, C//nheads] -> [bs*t, hw, C]
        memory = self.proj(memory)
        memory = self.proj_drop(memory)
        memory = rearrange(memory, '(b t) n c -> b (t n) c', b=B, t=T, n=hw)    # [bs, t*hw, C]

        # 2) query & memory: self-attention & cross-attention
        query_memory_k, query_memory_v = torch.cat([query_k, memory_k], dim=2), torch.cat([query_v, memory_v], dim=2)  # [bs, nheads, hw + t*hw, C//nheads]
        query_attn = (query_q @ query_memory_k.transpose(-2, -1)) * self.scale   # [bs, nheads, hw, hw + t*hw]
        query_attn = query_attn.softmax(dim=-1)
        query_attn = self.attn_drop(query_attn)

        query = (query_attn @ query_memory_v).transpose(1, 2).reshape(B, hw, C)  # [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]
        query = self.proj(query)
        query = self.proj_drop(query)

        # 3) cls_token & memory: self-attention & cross-attention
        if backbone_update:
            clstoken_memory_k, clstoken_memory_v = torch.cat([clstoken_k, memory_k], dim=2), torch.cat([clstoken_v, memory_v], dim=2)   # [bs, nheads, 1 + t*hw, C//nheads]
            clstoken_attn = (clstoken_q @ clstoken_memory_k.transpose(-2, -1)) * self.scale   # [bs, nheads, 1, 1 + t*hw]
            clstoken_attn = clstoken_attn.softmax(dim=-1)
            clstoken_attn = self.attn_drop(clstoken_attn)

            cls_token = (clstoken_attn @ clstoken_memory_v).transpose(1, 2).reshape(B, 1, C)  # [bs, nheads, 1, C//nheads] -> [bs, 1, nheads, C//nheads] -> [bs, 1, C]
            cls_token = self.proj(cls_token)
            cls_token = self.proj_drop(cls_token)
        else:
            cls_token = x[:, :1, :]

        x = torch.cat([cls_token, query, memory], dim=1)   # cls_token & query & memory [bs, 1 + hw + t*hw, 784]
        return x

    def clsToken_topk(self, clstoken_memory_attn, hw):
        """

        @param clstoken_memory_attn:[bs, nheads, 1, 1 + hw]
        @return:
            clstoken_memory_attn:[bs, nheads, 1, 1 + hw]
        """

        # softmax should along 1 clstoken and topK memory!!!

        # 1) split
        clstoken_attn, memory_attn = torch.split(clstoken_memory_attn, [1, hw], dim=3)  # [bs, nheads, 1, 1], [bs, nheads, 1, hw]

        # 2) select topK from each hw memory
        topK_values, topK_indices = torch.topk(input=memory_attn, k=self.topk_num, dim=3)  # [bs, nheads, 1, top_k]

        # 3) cat cls_token & topK
        clstoken_topK_attn = torch.cat([clstoken_attn, topK_values], dim=3)  # [bs, nheads, 1, 1 + topk_num]

        # 4) softmax
        clstoken_topK_attn = clstoken_topK_attn.softmax(dim=-1)

        # 5) split cls_token & topK
        clstoken_attn, topK_values = torch.split(clstoken_topK_attn, [1, self.topk_num], dim=3)

        # 6) input softmaxed T * topK into T * hw
        memory_affinity = torch.zeros_like(memory_attn).scatter_(3, topK_indices, topK_values.to(memory_attn.dtype))  # [bs, nheads, 1, hw]

        # 7) cat cls_token & memory
        clstoken_memory_attn = torch.cat([clstoken_attn, memory_affinity], dim=3)  # [bs, nheads, 1, 1 + hw]

        return clstoken_memory_attn

    def query_topk(self, query_memory_attn, hw, T):
        """
        normalize similarity with top-k softmax
        @param query_memory_attn: [bs, nheads, hw, N], N == hw + T * hw
        @return:
        """

        # softmax should along hw query and topK memory!!!

        query_attn, memory_attn = torch.split(query_memory_attn, [hw, T * hw], dim=3)  # [bs, nheads, hw, hw], [bs, nheads, hw, T * hw]
        if self.topk_range == 'every_frame':  # memory_attn [bs, nheads, hw, T * hw]
            # 1) reshape attention matrix as T memory frames
            memory_attn = rearrange(memory_attn, 'b h n1 (t n2) -> b h n1 t n2', n1=hw, t=T, n2=hw)  # [bs, nheads, hw, T, hw]

            # 2) select topK from each hw memory
            topK_values, topK_indices = torch.topk(input=memory_attn, k=self.topk_num, dim=4)  # [bs, nheads, hw, T, top_k]

            # 3) cat with query_attn, each query do softmax with [query : T*topK]
            topK_values = rearrange(topK_values, 'b h n1 t n2 -> b h n1 (t n2)', n1=hw, t=T, n2=self.topk_num)  # [bs, nheads, hw, T * topk_num]
            query_topK_attn = torch.cat([query_attn, topK_values], dim=3)  # [bs, nheads, hw, hw + T * topk_num]
            query_topK_attn = query_topK_attn.softmax(dim=-1)

            # 4) split into query and T * topK, input softmaxed T * topK into T * hw
            query_attn, topK_values = torch.split(query_topK_attn, [hw, T * self.topk_num], dim=3)
            topK_values = rearrange(topK_values, 'b h n1 (t n2) -> b h n1 t n2', n1=hw, t=T, n2=self.topk_num)  # [bs, nheads, hw, T, top_k]
            memory_affinity = torch.zeros_like(memory_attn).scatter_(4, topK_indices, topK_values.to(memory_attn.dtype))  # [bs, nheads, hw, T, hw]

            # 5) reshape back Thw, cat softmaxed hw query and softmaxed Thw memory
            memory_affinity = rearrange(memory_affinity, 'b h n1 t n2 -> b h n1 (t n2)', n1=hw, t=T, n2=hw)  # [bs, nheads, hw, T * hw]
            query_memory_attn = torch.cat([query_attn, memory_affinity], dim=3)  # [bs, nheads, hw, hw + T * hw]

            return query_memory_attn

        elif self.topk_range == 'all_frames':  # memory_attn [bs, nheads, hw, T * hw]
            # 1) select topK from all Thw memory
            topK_values, topK_indices = torch.topk(input=memory_attn, k=self.topk_num, dim=3)  # [bs, nheads, hw, top_k]

            # 2) cat with query_attn, each query do softmax with [query:topK]
            query_topK_attn = torch.cat([query_attn, topK_values], dim=3)  # [bs, nheads, hw, hw+top_k]
            query_topK_attn = query_topK_attn.softmax(dim=-1)

            # 3) split into query and topK, input softmaxed topK into Thw
            query_attn, topK_values = torch.split(query_topK_attn, [hw, self.topk_num], dim=3)
            memory_affinity = torch.zeros_like(memory_attn).scatter_(3, topK_indices, topK_values.to(memory_attn.dtype))  # [bs, nheads, hw, T * hw]

            # 4) cat softmaxed hw query and softmaxed Thw memory
            query_memory_attn = torch.cat([query_attn, memory_affinity], dim=3)  # [bs, nheads, hw, hw + T * hw]

            return query_memory_attn

        else:
            raise NotImplementedError

    def mixattn_memory_clsToken(self, x, hw, backbone_update_clsToken):
        """

        @param x: cls_token & memory [bs_obj, 1+hw, C] or memory [bs_obj, hw, C]
        @param hw:
        @param backbone_update_clsToken:
        @return: x
        """

        if backbone_update_clsToken:
            B, N, C = x.shape
            assert N == 1 + hw
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
            q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, nheads, N, C//nheads]
            clstoken_q, memory_q = torch.split(q, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
            clstoken_k, memory_k = torch.split(k, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
            clstoken_v, memory_v = torch.split(v, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]

            # 1) memory: self-attention
            memory_attn = (memory_q @ memory_k.transpose(-2, -1)) * self.scale  # [bs, nheads, hw, hw]
            memory_attn = memory_attn.softmax(dim=-1)
            memory_attn = self.attn_drop(memory_attn)

            memory = (memory_attn @ memory_v).transpose(1, 2).reshape(B, hw, C)  # [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]
            memory = self.proj(memory)
            memory = self.proj_drop(memory) # [bs, hw, C]

            # 2) cls_token: self-attention & cross-attention from memory
            clstoken_attn = (clstoken_q @ k.transpose(-2, -1)) * self.scale  # [bs, nheads, 1, 1 + hw]
            if self.topk_block and self.topk_clstoken:
                clstoken_attn = self.clsToken_topk(clstoken_memory_attn=clstoken_attn, hw=hw)
            else:
                clstoken_attn = clstoken_attn.softmax(dim=-1)
            clstoken_attn = self.attn_drop(clstoken_attn)

            cls_token = (clstoken_attn @ v).transpose(1, 2).reshape(B, 1, C)  # [bs, nheads, 1, C//nheads] -> [bs, 1, nheads, C//nheads] -> [bs, 1, C]
            cls_token = self.proj(cls_token)
            cls_token = self.proj_drop(cls_token)   # [bs, 1, C]

            # concatenate them
            x = torch.cat([cls_token, memory], dim=1)   # [bs, 1+hw, C]
            return x
        else:
            B, N, C = x.shape
            assert N == hw
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
            q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, nheads, N, C//nheads]

            # memory: self-attention
            memory_attn = (q @ k.transpose(-2, -1)) * self.scale  # [bs, nheads, hw, hw]
            memory_attn = memory_attn.softmax(dim=-1)
            memory_attn = self.attn_drop(memory_attn)

            memory = (memory_attn @ v).transpose(1, 2).reshape(B, hw, C)  # [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]
            memory = self.proj(memory)
            memory = self.proj_drop(memory)  # [bs, hw, C]

            return memory

    def mixattn_memory_query(self, x, T, hw):
        """
        readout memory: query <- [query: ref memorys] in block3
        @param x: query&memory [bs_obj, hw+Thw, C]
        @return:
            query&memory [bs_obj, hw+Thw, C], not update memory part
        """
        B, N, C = x.shape
        assert N == hw + T*hw
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, nheads, N, C//nheads]
        query_q, memory_q = torch.split(q, [hw, T * hw], dim=2)  # [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]
        query_k, memory_k = torch.split(k, [hw, T * hw], dim=2)  # [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]
        query_v, memory_v = torch.split(q, [hw, T * hw], dim=2)  # [bs, nheads, hw, C//nheads], [bs, nheads, t*hw, C//nheads]

        # query: self-attention & cross-attention from memory
        query_attn = (query_q @ k.transpose(-2, -1)) * self.scale  # [bs, nheads, hw, hw + t*hw]
        if self.topk_block:
            query_attn = self.query_topk(query_memory_attn=query_attn, hw=hw, T=T)
        else:
            query_attn = query_attn.softmax(dim=-1)
        query_attn = self.attn_drop(query_attn)

        query = (query_attn @ v).transpose(1, 2).reshape(B, hw, C)  # [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]
        query = self.proj(query)
        query = self.proj_drop(query)

        _, memory = torch.split(x, [hw, T*hw], dim=1)
        x = torch.cat([query, memory], dim=1)

        return x

    def enhance_query(self, x, hw):
        """
        enhance query with cls_token in block5
        @param x: cls_token & f16 [bs_obj, 1+hw, C]
        @return:
            cls_token & f16 [bs_obj, 1+hw, C]
        """
        assert self.topk_block is False and self.topk_num is None and self.topk_range == 'None'
        cls_token, query = torch.split(x, [1, hw], dim=1)

        B, N, C = x.shape
        assert N == 1 + hw
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, nheads, N, C//nheads]
        clstoken_q, query_q = torch.split(q, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
        clstoken_k, query_k = torch.split(k, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
        clstoken_v, query_v = torch.split(v, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]

        attn = (query_q @ clstoken_k.transpose(-2, -1)) * self.scale  # [bs, nheads, hw, 1]
        attn = torch.sigmoid(attn)  # softmax -> sigmoid
        attn = self.attn_drop(attn)

        # score on query spatial
        enhanced_query = (attn @ clstoken_v).transpose(1, 2).reshape(B, hw, C)    # [bs, nheads, hw, 1]@[bs, nheads, 1, C//nheads] = [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]

        enhanced_query = self.proj(enhanced_query)
        enhanced_query = self.proj_drop(enhanced_query)

        assert query.shape == enhanced_query.shape
        x = torch.cat([cls_token, enhanced_query], dim=1)

        return x

    def mixattn_feature_clstoken(self, x, hw):
        """
        update cls_token with feature in block4
        @param x: cls_token & feature [bs_obj, 1+hw, C]
        @param hw:
        @return:
            x: cls_token & feature [bs_obj, 1+hw, C]
        """
        B, N, C = x.shape
        assert N == 1 + hw
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, bs, nheads, N, C//nheads]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, nheads, N, C//nheads]
        clstoken_q, feature_q = torch.split(q, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
        clstoken_k, feature_k = torch.split(k, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]
        clstoken_v, feature_v = torch.split(v, [1, hw], dim=2)  # [bs, nheads, 1, C//nheads], [bs, nheads, hw, C//nheads]

        # 1) feature: self-attention
        feature_attn = (feature_q @ feature_k.transpose(-2, -1)) * self.scale  # [bs, nheads, hw, hw]
        feature_attn = feature_attn.softmax(dim=-1)
        feature_attn = self.attn_drop(feature_attn)

        feature = (feature_attn @ feature_v).transpose(1, 2).reshape(B, hw, C)  # [bs, nheads, hw, C//nheads] -> [bs, hw, nheads, C//nheads] -> [bs, hw, C]
        feature = self.proj(feature)
        feature = self.proj_drop(feature)  # [bs, hw, C]

        # 2) cls_token: self-attention & cross-attention from feature
        clstoken_attn = (clstoken_q @ k.transpose(-2, -1)) * self.scale  # [bs, nheads, 1, 1 + hw]
        if self.topk_block and self.topk_clstoken:
            clstoken_attn = self.clsToken_topk(clstoken_memory_attn=clstoken_attn, hw=hw)
        else:
            clstoken_attn = clstoken_attn.softmax(dim=-1)
        clstoken_attn = self.attn_drop(clstoken_attn)

        cls_token = (clstoken_attn @ v).transpose(1, 2).reshape(B, 1, C)  # [bs, nheads, 1, C//nheads] -> [bs, 1, nheads, C//nheads] -> [bs, 1, C]
        cls_token = self.proj(cls_token)
        cls_token = self.proj_drop(cls_token)  # [bs, 1, C]

        # concatenate them
        x = torch.cat([cls_token, feature], dim=1)  # [bs, 1+hw, C]
        return x

    def forward(self, mode, *args, **kwargs):
        if mode == 'mixattn_memory_clsToken':
            return self.mixattn_memory_clsToken(*args, **kwargs)
        elif mode == 'mixattn_memory_query':
            return self.mixattn_memory_query(*args, **kwargs)
        elif mode == 'enhance_query':
            return self.enhance_query(*args, **kwargs)
        elif mode == 'mixattn_feature_clstoken':
            return self.mixattn_feature_clstoken(*args, **kwargs)
        else:
            raise NotImplementedError


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 topk_num=None, topk_range='every_frame', topk_block=False, topk_clstoken=False
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            topk_num=topk_num, topk_range=topk_range, topk_block=topk_block, topk_clstoken=topk_clstoken
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_feature(self, x, hw, t, backbone_update):
        x = x + self.drop_path(self.attn(self.norm1(x), hw, t, backbone_update))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def mixattn_memory_clsToken(self, x, hw, backbone_update_clsToken):
        x = x + self.drop_path(self.attn('mixattn_memory_clsToken', self.norm1(x), hw, backbone_update_clsToken))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def mixattn_memory_query(self, x, T, hw):
        x = x + self.drop_path(self.attn('mixattn_memory_query', self.norm1(x), T, hw))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def enhance_query(self, x, hw):
        x = x + self.drop_path(self.attn('enhance_query', self.norm1(x), hw))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def mixattn_feature_clstoken(self, x, hw):
        x = x + self.drop_path(self.attn('mixattn_feature_clstoken', self.norm1(x), hw))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, mode, *args, **kwargs):
        if mode == 'mixattn_memory_clsToken':
            return self.mixattn_memory_clsToken(*args, **kwargs)
        elif mode == 'mixattn_memory_query':
            return self.mixattn_memory_query(*args, **kwargs)
        elif mode == 'enhance_query':
            return self.enhance_query(*args, **kwargs)
        elif mode == 'mixattn_feature_clstoken':
            return self.mixattn_feature_clstoken(*args, **kwargs)
        else:
            raise NotImplementedError

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
    def forward(self, x):
        B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, single_object=False,
                 update_block_depth=2, enhance_block_depth=1,
                 topk_num=None, topk_range='every_frame', topk_block='all_blocks', topk_clstoken='N',
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.single_object = single_object

        self.topk_num, self.topk_range, self.topk_clstoken = topk_num, topk_range, topk_clstoken
        if topk_block == 'last_block':
            self.topk_block = [depth[2] - 1]
        elif topk_block == 'all_blocks':
            self.topk_block = list(range(depth[2]))
        elif topk_block == 'None':
            self.topk_block = []
        elif topk_block == 'half':
            self.topk_block = list(range(depth[2] // 2, depth[2]))
        else:
            raise NotImplementedError

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed1_query = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed1_memory = PatchEmbed(
                img_size=img_size[0], patch_size=patch_size[0], in_chans=(in_chans + 1) if single_object else (in_chans + 2), embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        num_patches = self.patch_embed3.num_patches
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim[2]))    # add cls_token
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer,
                topk_num=self.topk_num, topk_range=self.topk_range, topk_block=(i in self.topk_block), topk_clstoken=(topk_clstoken == 'Y')
            )
            for i in range(depth[2])])

        self.patch_embed4_block4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.blocks4 = nn.ModuleList([  # cls_token <- [cls_token: fusion_feature]
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                topk_num=self.topk_num, topk_range=self.topk_range, topk_block=(len(self.topk_block)!=0), topk_clstoken=(topk_clstoken == 'Y')
            )
            for i in range(update_block_depth)])
        self.update_block_depth = update_block_depth

        self.patch_embed4_block5 = nn.Linear(embed_dim[2], embed_dim[2])
        self.blocks5 = nn.ModuleList([  # query <- cls_token
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
                topk_num=None, topk_range='None', topk_block=False, topk_clstoken=False
            )
            for i in range(enhance_block_depth)])
        self.enhance_block_depth = enhance_block_depth

        self.norm = norm_layer(embed_dim[-1])

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[2]))

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        self.load_pretrained(pretrained='./checkpoints/convmae_base.pth')
        self._initialize_pos_weights()
        self._initialize_cls_token()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        # pretrained = pretrained or self.pretrained

        if isinstance(pretrained, str) and pretrained != '':
            self.apply(self._init_weights)

            print(f"load from {pretrained}")
            ckpt = torch.load(pretrained, map_location='cpu')
            if 'model' in ckpt.keys():
                ckpt = ckpt['model']

            new_dict = {}
            unused_keys = list()

            for k, v in ckpt.items():
                # new_dict[k] = v
                if k == 'pos_embed':
                    unused_keys.append(k)
                elif k == 'patch_embed1.proj.weight':
                    new_dict['patch_embed1_query.proj.weight'] = v
                    pads = torch.zeros((256, 1, 4, 4), device=v.device) if self.single_object else torch.zeros((256, 2, 4, 4), device=v.device)
                    trunc_normal_(pads, std=0.02)
                    v = torch.cat([v, pads], 1)
                    new_dict['patch_embed1_memory.proj.weight'] = v
                elif 'patch_embed1' in k:
                    new_dict['patch_embed1_query' + k[12:]] = v
                    new_dict['patch_embed1_memory' + k[12:]] = v
                elif 'patch_embed4' in k:
                    new_dict[k] = v
                    new_dict['patch_embed4_block4' + k[len('patch_embed4'):]] = v
                    new_dict['patch_embed4_block5' + k[len('patch_embed4'):]] = v
                elif 'blocks3' in k:
                    new_dict[k] = v
                    _, depth, module = k.split('.', 2)
                    depth = int(depth)
                    # blocks4
                    differ = 11 - self.update_block_depth
                    if depth - differ >= 0:
                        new_dict[f'blocks4.{depth - differ}.' + module] = v
                    # blocks5
                    differ = 11 - self.enhance_block_depth
                    if depth - differ >= 0:
                        new_dict[f'blocks5.{depth - differ}.' + module] = v
                else:
                    new_dict[k] = v

            missing_keys, unexpected_keys = self.load_state_dict(new_dict, strict=False)
            assert unexpected_keys == []
            print("unusing keys:", unused_keys)  # filter key which we don't want to use
            print("missing keys:", missing_keys)  # model's keys, but checkpoint not support
            print("unexpected keys:", unexpected_keys)  # checkpoint support, but not in model
            print("Loading pretrained ViT done.")

            return

        elif pretrained is None or str == '':
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _initialize_pos_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        num_patches = self.pos_embed.shape[1]
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _initialize_cls_token(self):
       torch.nn.init.normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, query, memory, cls_token, backbone_update):
        """

        @param query: frame[bs, 3, H, W]
        @param memory: ref_frames_with_masks[bs, T, 3+1 or 3+2, H, W]
        @param cls_token: [bs, N=1, C]
        @param backbone_update: if True, cls_token do cross-attention in mix-attn block
        @return:
            f16, f8, f4 [bs, c, h, w]
            cls_token: [bs, N=1, C]
        """

        features = []
        B, T, C, H, W = memory.shape
        memory = rearrange(memory, 'b t c h w -> (b t) c h w')  # [bs*t, c, h, w]

        query, memory = self.patch_embed1_query(query), self.patch_embed1_memory(memory)  # [bs, 256, 96, 96], [bs*t, 256, 96, 96]
        query, memory = self.pos_drop(query), self.pos_drop(memory)
        for blk in self.blocks1:
            query = blk(query)
            memory = blk(memory)
        f4 = query.contiguous() # [bs, 256, 96, 96]

        query, memory = self.patch_embed2(query), self.patch_embed2(memory)  # [bs, 384, 48, 48], [bs*t, 384, 48, 48]
        for blk in self.blocks2:
            query = blk(query)
            memory = blk(memory)
        f8 = query.contiguous() # [bs, 384, 48, 48]

        query, memory = self.patch_embed3(query), self.patch_embed3(memory) # [bs, 784, 24, 24], [bs*t, 784, 24, 24]
        h, w = query.shape[2:]
        query, memory = query.flatten(2).permute(0, 2, 1), memory.flatten(2).permute(0, 2, 1)  # [bs, 24*24, 784], [bs*t, 24*24, 784]
        query, memory = self.patch_embed4(query), self.patch_embed4(memory) # [bs, 24*24, 784], [bs*t, 24*24, 784]

        # positional encoding
        # query, memory = query + self.pos_embed[:, 1:, :], memory + self.pos_embed[:, 1:, :]
        query = query + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])
        memory = memory + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])
        cls_token = cls_token + self.pos_embed[:, :1, :]

        # Mix-attn block
        memory = rearrange(memory, '(b t) n c -> b (t n) c', b=B, t=T)  # [bs*t, 24*24, 784] -> [bs, t * 24*24, 784]
        clstoken_query_memory = torch.cat([cls_token, query, memory], dim=1)    # [bs, 1 + 24*24 + t * 24*24, 784]
        for blk in self.blocks3:
            clstoken_query_memory = blk(x=clstoken_query_memory, hw=h*w, t=T, backbone_update=backbone_update)    # [bs, 24*24 + t * 24*24, 784]
        cls_token, query, memory = torch.split(clstoken_query_memory, [1, h*w, T*h*w], dim=1) # [bs, 1, 784], [bs, 24*24, 784], [bs, t * 24*24, 784]
        memory = rearrange(memory, 'b (t n) c -> (b t) n c', b=B, t=T)  # [bs, t * 24*24, 784] -> [bs*t, 24*24, 784]
        query, memory = self.norm(query), self.norm(memory)
        query, memory = query.permute(0, 2, 1).reshape(B, -1, h, w), memory.permute(0, 2, 1).reshape(B*T, -1, h, w) # [bs, 784, 24, 24], [bs*t, 784, 24, 24]
        f16 = query.contiguous()    # [bs, 784, 24, 24]

        return [f16, f8, f4], cls_token

    def mixattn_memory_clsToken(self, memory, cls_token, backbone_update_clsToken):
        """
        get block value for this memory frame&mask, and updated cls_token in block3 if backbone_update_clsToken
        @param memory: ref_frames_with_masks[bs_obj, 3+1 or 3+2, H, W]
        @param cls_token: [bs_obj, N=1, C=768]
        @return:
            block_value: {
                    f4, f8, f16: [bs_obj, c, h, w]
                    block{index}: [bs_obj, hw, C]
                }
            cls_token: [bs_obj, N=1, C=768]
        """

        block_value = dict()

        B, C, H, W = memory.shape
        memory = self.patch_embed1_memory(memory)   # [bs_obj, 256, h4, w4]
        memory = self.pos_drop(memory)
        for blk in self.blocks1:
            memory = blk(memory)
        block_value['f4'] = memory.contiguous()   # [bs_obj, 256, h4, w4]

        memory = self.patch_embed2(memory)  # [bs_obj, 384, h8, w8]
        for blk in self.blocks2:
            memory = blk(memory)
        block_value['f8'] = memory.contiguous()  # [bs_obj, 384, h8, w8]

        memory = self.patch_embed3(memory)  # [bs_obj, 768, h16, w16]
        h, w = memory.shape[2:]
        memory = memory.flatten(2).permute(0, 2, 1) # [bs_obj, hw, 768]
        memory = self.patch_embed4(memory)  # NOTE: cls_token no need patch_embed4

        if backbone_update_clsToken:
            # positional encoding
            memory = memory + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])
            cls_token = cls_token + self.pos_embed[:, :1, :]

            # mix-attn block3
            clstoken_memory = torch.cat([cls_token, memory], dim=1) # [bs_obj, 1+hw, 768]
            for idx, blk in enumerate(self.blocks3):
                block_value[f'block_value{idx}'] = clstoken_memory[:, 1:]
                clstoken_memory = blk('mixattn_memory_clsToken', x=clstoken_memory, hw=h*w, backbone_update_clsToken=backbone_update_clsToken)
            cls_token, memory = torch.split(clstoken_memory, [1, h*w], dim=1)

            cls_token, memory = self.norm(cls_token), self.norm(memory)
        else:
            # positional encoding
            memory = memory + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])

            # mix-attn block3
            for idx, blk in enumerate(self.blocks3):
                block_value[f'block_value{idx}'] = memory
                memory = blk('mixattn_memory_clsToken', x=memory, hw=h*w, backbone_update_clsToken=backbone_update_clsToken)

            memory = self.norm(memory)

        memory = memory.permute(0, 2, 1).reshape(B, -1, h, w)   # [bs_obj, c, h, w]
        block_value['f16'] = memory.contiguous()

        return block_value, cls_token

    def mixattn_memory_query(self, query, memory_block_values):
        """
        readout memory: query <- [query: ref memorys] in block3
        @param query: [bs_obj, 3, H, W]
        @param memory_block_values: {
                f4, f8, f16: [bs_obj, T, c, h, w]
                block{index}: [bs_obj, T, hw, C]
            }
        @return:
            f16, f8, f4 [bs_obj, c, h, w]
        """

        B, T, C, h16, w16 = memory_block_values['f16'].shape

        query = self.patch_embed1_query(query)  # [bs_obj, 256, h4, w4]
        query = self.pos_drop(query)
        for blk in self.blocks1:
            query = blk(query)
        f4 = query.contiguous() # [bs_obj, 256, h4, w4]

        query = self.patch_embed2(query)  # [bs_obj, 384, h8, w8]
        for blk in self.blocks2:
            query = blk(query)
        f8 = query.contiguous() # [bs_obj, 384, h8, w8]

        query = self.patch_embed3(query)    # [bs_obj, 768, h16, w16]
        h, w = query.shape[2:]
        query = query.flatten(2).permute(0, 2, 1)   # [bs_obj, hw, 768]
        query = self.patch_embed4(query)

        # positional encoding
        query = query + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])

        # Mix-attn block
        for idx, blk in enumerate(self.blocks3):
            memory = memory_block_values[f'block_value{idx}']  # [bs_obj, T, hw, C]
            memory = rearrange(memory, 'b t n c -> b (t n) c', b=B, t=T, n=h*w, c=self.embed_dim[-1])    # [bs_obj, Thw, C]
            query_memory = torch.cat([query, memory], dim=1)    # [bs_obj, hw+Thw, C]
            query_memory = blk('mixattn_memory_query', x=query_memory, T=T, hw=h*w)
            query, _ = torch.split(query_memory, [h*w, T*h*w], dim=1)

        query = self.norm(query)
        query = query.permute(0, 2, 1).reshape(B, -1, h, w) # [bs_obj, C, h, w]
        f16 = query.contiguous()  # [bs,  C, h, w]

        return [f16, f8, f4]

    def enhance_query(self, f16, cls_token):
        """
        enhance query with cls_token in block5
        @param f16: [bs_obj, C=768, h, w]
        @param cls_token: [bs_obj, N=1, C=768]
        @return:
            enhanced_f16: [bs_obj, C=768, h, w]
        """

        B, C, h, w = f16.shape

        f16 = f16.flatten(2).permute(0, 2, 1)    # [bs_obj, hw, C]

        cls_token, f16 = self.patch_embed4_block5(cls_token), self.patch_embed4_block5(f16)

        cls_token = cls_token + self.pos_embed[:, :1, :]
        f16 = f16 + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])

        clstoken_f16 = torch.cat([cls_token, f16], dim=1)   # [bs_obj, 1+ hw, C]

        for blk in self.blocks5:
            clstoken_f16 = blk('enhance_query', x=clstoken_f16, hw=h*w)

        cls_token, enhanced_f16 = torch.split(clstoken_f16, [1, h*w], dim=1)

        cls_token, enhanced_f16 = self.norm(cls_token), self.norm(enhanced_f16)

        enhanced_f16 = enhanced_f16.permute(0, 2, 1).reshape(B, -1, h, w) # [bs_obj, C, h, w]

        return enhanced_f16

    def mixattn_feature_clstoken(self, fusion_feature, cls_token):
        """
        update cls_token <- [cls_token: fusion_feature] in convmae.block4
        @param fusion_feature: [bs_obj, C=768, h, w]
        @param cls_token: [bs_obj, N=1, C=768]
        @return:
            cls_token: [bs_obj, N=1, C=768]
        """
        B, C, h, w = fusion_feature.shape
        fusion_feature = fusion_feature.flatten(2).permute(0, 2, 1) # [bs_obj, hw, C]

        cls_token, fusion_feature = self.patch_embed4_block4(cls_token), self.patch_embed4_block4(fusion_feature)

        cls_token = cls_token + self.pos_embed[:, :1, :]
        fusion_feature = fusion_feature + get_abs_pos(abs_pos=self.pos_embed[:, 1:, :], hw=(h, w), has_cls_token=False).reshape(1, h * w, self.embed_dim[-1])

        clstoken_feature = torch.cat([cls_token, fusion_feature], dim=1)    # [bs_obj, 1+hw, C]
        for blk in self.blocks4:
            clstoken_feature = blk('mixattn_feature_clstoken', x=clstoken_feature, hw=h*w)
        cls_token, fusion_feature = torch.split(clstoken_feature, [1, h*w], dim=1)

        cls_token = self.norm(cls_token)

        return cls_token

    def forward(self, mode, *args, **kwargs):
        if mode == 'mixattn_memory_clsToken':
            return self.mixattn_memory_clsToken(*args, **kwargs)
        elif mode == 'mixattn_memory_query':
            return self.mixattn_memory_query(*args, **kwargs)
        elif mode == 'enhance_query':
            return self.enhance_query(*args, **kwargs)
        elif mode == 'mixattn_feature_clstoken':
            return self.mixattn_feature_clstoken(*args, **kwargs)
        else:
            raise NotImplementedError


def convvit_base_patch16(single_object, config):
    # assert 'topk_num' in config.keys()
    topk_num = config.get('topk_num', None)
    topk_range = config.get('topk_range', 'every_frame')
    topk_block = config.get('topk_block', 'None')
    topk_clstoken = config.get('topk_clstoken', 'N')
    model = ConvViT(
        img_size=[384, 96, 48], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), single_object=single_object,
        update_block_depth=config['update_block_depth'], enhance_block_depth=config['enhance_block_depth'],
        topk_num=topk_num, topk_range=topk_range, topk_block=topk_block, topk_clstoken=topk_clstoken
    )
    return model

if __name__ == '__main__':
    model = convvit_base_patch16().cuda()
    img = torch.randn((2, 3, 384, 384)).cuda()
    a = model(img)
