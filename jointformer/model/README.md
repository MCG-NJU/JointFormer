
## conmae.py:

1) patch_embed1_memory 的 conv input_channel为5（随机初始化），之后memory_feat和query_feat一样
1) mix-attention:
    1) self-attn: T memory frame do self-attention independent
    1) mix-attn:
        1) query frame as q, [query frame: T memory frames] as k,
        1) attention matrix softmax along all key
        1) attention matrix @ [query frame: T memory frames] as v
1) convmae return: 未过block的f4(256)和f8(384)，已经过block的f16(768)
1) decoder: f16直接变为g16（没有img_feat16)，然后和f8&f4执行U-Net



## conmae_topK.py:

1) patch_embed1_memory 的 conv input_channel为5（随机初始化），之后memory_feat和query_feat一样
1) mix-attention:
    1) self-attn: T memory frame do self-attention independent
    1) mix-attn:
        1) query frame as q, [query frame: T memory frames] as k/v,
        1) each query choose query frame and "topK" memory in all/each memory frame, attention matrix softmax along them, set else attention similarity ZERO
        1) attention matrix @ [query frame: T memory frames] as v, since "not topK" memory's attention similarity are zero, only value from query frame and "topK" memory are used 
1) convmae return: 未过block的f4(256)和f8(384)，已经过block的f16(768)
1) decoder: f16直接变为g16（没有img_feat16)，然后和f8&f4执行U-Net



## conmae_usage.py:

1) patch_embed1_memory 的 conv input_channel为5（随机初始化），之后memory_feat和query_feat一样
1) mix-attention:
    1) self-attn: T memory frame do self-attention independent
    1) mix-attn:
        1) query frame as q, [query frame: T memory frames] as k,
        1) attention matrix softmax along all key
        1) attention matrix @ [query frame: T memory frames] as v
    1) mix-attn attention matrix[bs, nhead, n_q=hw, n_q=hw + n_m=T*hw]
        1) split -> [bs, nhead, n_q=hw, n_m=T*hw]
        1) mean along all nhead -> [bs, n_q=hw, n_m=T*hw]
        1) split each memory frame -> [bs, T, n_q=hw, n_m=hw]
        1) sum along all keys and all querys -> [bs, T]
1) convmae return: 未过block的f4(256)和f8(384)，已经过block的f16(768)
1) decoder: f16直接变为g16（没有img_feat16)，然后和f8&f4执行U-Net