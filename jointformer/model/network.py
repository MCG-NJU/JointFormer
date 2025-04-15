"""
This file defines JointFormer, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

from jointformer.model.convmae_v1 import convvit_base_patch16
from jointformer.model.modules import Decoder
from jointformer.model.aggregate import aggregate

from timm.models.layers import trunc_normal_

class JointFormer(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        self.convmae = convvit_base_patch16(single_object=self.single_object, config=config)

        self.decoder = Decoder(f16_dim=768, f8_dim=384, f4_dim=256)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_value(self, frame, masks):
        """
        concatenate frame and mask
        @param frame: [bs, 3, H, W]
        @param masks: [bs, max_obj_num, H, W]
        @return:
        """
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        if self.single_object:
            masks = masks.unsqueeze(2) # [bs, max_obj_num, 1, H, W]
        else:
            masks = torch.stack([masks, others], 2) # [bs, max_obj_num, 2, H, W]

        frames = frame.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) # [bs, max_obj_num, 3, H, W]
        frames_with_masks = torch.cat([frames, masks], 2)   # [bs, max_obj_num, 3+1 or 3+2, H, W]

        return frames_with_masks

    def mixattn_memory_clsToken(self, memory, cls_token, backbone_update_clsToken):
        """
        get block value for this memory frame&mask, and updated cls_token in block3 if backbone_update_clsToken
        @param memory: [bs, max_obj_num, 3+1 or 3+2, H, W]
        @param cls_token: [bs, max_obj_num, N=1, C=768]
        @param backbone_update_clsToken: if Ture, cls_token updated in mix-attn block3 with this memory
        @return:
            block_value: {
                f4, f8, f16: [bs, max_obj_num, c, h, w]
                block{index}: [bs, max_obj_num, hw, C]
            }
            cls_token: [bs, max_obj_num, N=1, C=768]
        """
        assert cls_token.shape[:2] == memory.shape[:2]

        bs, num_objects = memory.shape[:2]

        cls_token = cls_token.reshape(bs*num_objects, *cls_token.shape[2:]) # [bs_obj, N=1, C]
        memory = memory.reshape(bs*num_objects, *memory.shape[2:]) # [bs_obj, 3+1 or 3+2, H, W]

        if backbone_update_clsToken:
            block_value, cls_token = self.convmae('mixattn_memory_clsToken',
                                                  memory=memory, cls_token=cls_token,
                                                  backbone_update_clsToken=backbone_update_clsToken)
        else:
            block_value, _ = self.convmae('mixattn_memory_clsToken',
                                          memory=memory, cls_token=cls_token,
                                          backbone_update_clsToken=backbone_update_clsToken)

        cls_token = cls_token.reshape(bs, num_objects, *cls_token.shape[1:])  # [bs_obj, N=1, C]
        for k in block_value.keys():
            value = block_value[k]  # [bs_obj, c, h, w] or [bs_obj, hw, C]
            value = value.reshape(bs, num_objects, *value.shape[1:]) #   # [bs, max_obj_num, c, h, w] or [bs, max_obj_num, hw, C]
            block_value[k] = value

        return block_value, cls_token

    def mixattn_memory_query(self, query_frame, ref_block_values:dict):
        """
        readout memory: query <- [query: ref memorys] in block3
        @param query_frame: [bs, 3, H, W]
        @param ref_block_values: {
                f4, f8, f16: [bs, max_obj_num, T, c, h, w]
                block{index}: [bs, max_obj_num, T, hw, C]
            }
        @return:
            f16, f8, f4 [bs, max_obj_num, c, h, w]
        """

        bs, num_objects = ref_block_values['f16'].shape[:2]
        query_frame = query_frame.unsqueeze(1).repeat(1, num_objects, 1, 1, 1)  # [bs, max_obj_num, 3, H, W]

        query_frame = query_frame.reshape(bs * num_objects, *query_frame.shape[2:])  # [bs_obj, 3, H, W]
        for k in ref_block_values.keys():
            value = ref_block_values[k]
            value = value.reshape(bs * num_objects, *value.shape[2:])
            ref_block_values[k] = value

        [f16, f8, f4] = self.convmae('mixattn_memory_query', query=query_frame, memory_block_values=ref_block_values)

        f16 = f16.reshape(bs, num_objects, *f16.shape[1:])
        f8 = f8.reshape(bs, num_objects, *f8.shape[1:])
        f4 = f4.reshape(bs, num_objects, *f4.shape[1:])

        return [f16, f8, f4]

    def enhance_query(self, f16, cls_token):
        """
        enhance query with cls_token in block5
        @param f16: [bs, max_obj_num, C=768, h, w]
        @param cls_token: [bs, max_obj_num, N=1, C=768]
        @return:
            enhanced_f16: [bs, max_obj_num, C=768, h, w]
        """

        assert f16.shape[:2] == cls_token.shape[:2]

        bs, num_objects = f16.shape[:2]
        f16 = f16.reshape(bs * num_objects, *f16.shape[2:])  # [bs_obj, C=768, h, w]
        cls_token = cls_token.reshape(bs * num_objects, *cls_token.shape[2:])  # [bs_obj, N=1, C=768]

        enhanced_f16 = self.convmae('enhance_query', f16=f16, cls_token=cls_token)  # [bs_obj, C=768, h, w]

        enhanced_f16 = enhanced_f16.reshape(bs, num_objects, *enhanced_f16.shape[1:])  # [bs, max_obj_num, C=768, h, w]

        return enhanced_f16

    def segment(self, multi_scale_features, enhanced_f16, selector=None, strip_bg=True):
        """
        get mask prediction with multi_scale_features and update_f16
        @param multi_scale_features: f16, f8, f4, [bs, max_obj_num, c, h, w]
        @param enhanced_f16: [bs, max_obj_num, c, h, w]
        @param selector:
        @param strip_bg:
        @return:
            fusion_feature: [bs, obj_num, 256*3=768, h16, w16]
            logits: [bs, 1+max_obj_num, H, W]
            prob: softmaxed logits [bs, 1+max_obj_num, H, W], or [bs, max_obj_num, H, W] if strip_bg=True
        """

        logits, fusion_feature = self.decoder(multi_scale_features, enhanced_f16)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return fusion_feature, logits, prob

    def mixattn_feature_clstoken(self, fusion_feature, cls_token):
        """
        update cls_token <- [cls_token: fusion_feature] in convmae.block4
        @param fusion_feature:  [bs, obj_num, 256*3=768, h16, w16]
        @param cls_token: [bs, obj_num, N=1, C=768]
        @return:
            cls_token: [bs, obj_num, N=1, C=768]
        """
        bs, num_objects = fusion_feature.shape[:2]
        fusion_feature = fusion_feature.reshape(bs * num_objects, *fusion_feature.shape[2:])  # [bs_obj, C=768, h, w]
        cls_token = cls_token.reshape(bs * num_objects, *cls_token.shape[2:])  # [bs_obj, N=1, C=768]

        cls_token = self.convmae('mixattn_feature_clstoken', fusion_feature=fusion_feature, cls_token=cls_token)

        cls_token = cls_token.reshape(bs, num_objects, *cls_token.shape[1:])  # [bs, max_obj_num, C=768, h, w]

        return cls_token

    def get_cls_token(self):
        return self.convmae.cls_token

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            raise NotImplementedError
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'mixattn_memory_clsToken':
            return self.mixattn_memory_clsToken(*args, **kwargs)
        elif mode == 'mixattn_memory_query':
            return self.mixattn_memory_query(*args, **kwargs)
        elif mode == 'enhance_query':
            return self.enhance_query(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'mixattn_feature_clstoken':
            return self.mixattn_feature_clstoken(*args, **kwargs)
        elif mode == 'get_cls_token':
            return self.get_cls_token(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        if model_path is not None:
            model_weights = torch.load(model_path, map_location=map_location)
            config['update_block_depth'] = len(set([a.split(".")[2] for a in model_weights.keys() if 'convmae.blocks4.' in a]))
            config['enhance_block_depth'] = len(set([a.split(".")[2] for a in model_weights.keys() if 'convmae.blocks5.' in a]))
            print(f"Hyperparameters read from the model weights: "
                    f"update_block_depth={config['update_block_depth']}, "
                    f"enhance_block_depth={config['enhance_block_depth']}")
        else:
            model_weights = None
            if 'update_block_depth' not in config:
                config['update_block_depth'] = 2
                print(f"update_block_depth not in config, set to default {config['update_block_depth']}")
            else:
                print(f"update_block_depth in config, set to {config['update_block_depth']}")
            
            if 'enhance_block_depth' not in config:
                config['enhance_block_depth'] = 1
                print(f"enhance_block_depth not in config, set to default {config['enhance_block_depth']}")
            else:
                print(f"enhance_block_depth in config, set to {config['enhance_block_depth']}")
        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        assert (self.single_object == False)
        k = 'convmae.patch_embed1_memory.proj.weight'
        if src_dict[k].shape[1] == 4:
            print('Converting weights from single object to multiple objects.')
            pads = torch.zeros((256, 1, 4, 4), device=src_dict[k].device)
            trunc_normal_(pads, std=0.02)
            # if not init_as_zero_if_needed:
            #     print('Randomly initialized padding.')
            #     nn.init.orthogonal_(pads)
            # else:
            #     print('Zero-initialized padding.')
            src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict, strict=True)
