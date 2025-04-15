import torch
import warnings
import copy
from typing import List

from jointformer.inference.memory_store import MemoryStore
from jointformer.model.network import JointFormer


class MemoryManager:
    def __init__(self, config, network:JointFormer):
        self.config = config

        self.include_last = config['include_last']
        self.max_memory_frames = config['max_memory_frames']

        self.network = network

        self.cls_token = None   # [obj_num, N=1, C=768]
        self.memory_store = MemoryStore(config)

    def match_memory(self, frame):
        """

        @param frame: [bs=1, 3, H, W]
        @return:
            [f16_list, f8_list, f4_list]: [bs=1, obj_num, c, h, w]
        """
        assert (self.memory_store.all_objects == list(self.memory_store.mem_frames_with_masks.keys()))

        f16_list, f8_list, f4_list = list(), list(), list()

        for obj in self.memory_store.all_objects:
            # ref_frames_with_masks
            frames_with_masks = self.memory_store.mem_frames_with_masks[obj]  # [T, 3+2, H, W]
            # add last frame
            if (self.include_last) and (obj in self.memory_store.tmp_mem_frames_with_masks.keys()):
                frames_with_masks = torch.cat(  # [T+1, 3+2, H, W]
                    [frames_with_masks, self.memory_store.tmp_mem_frames_with_masks[obj]], dim=0
                )
            # add batch_size dimension
            frames_with_masks = frames_with_masks.unsqueeze(0).unsqueeze(0) # [bs=1, obj_num=1, T, 3+2, H, W]

            # ref_block_values
            memory_value = copy.deepcopy(self.memory_store.memory_value[obj])  # [T, c, h, w] or [T, hw, c]
            # add last frame
            if (self.include_last) and (obj in self.memory_store.tmp_memory_value.keys()):
                tmp_memory_value = copy.deepcopy(self.memory_store.tmp_memory_value[obj])  # [T=1, c, h, w] or [T=1, hw, c]
                assert memory_value.keys() == tmp_memory_value.keys()
                for k in memory_value.keys():
                    memory_value[k] = torch.cat(  # [T+1, c, h, w] or [T+1, hw, c]
                        [memory_value[k], tmp_memory_value[k]], dim=0
                    )
            # add batch_size dimension
            for k in memory_value.keys():
                memory_value[k] = memory_value[k].unsqueeze(0).unsqueeze(0) # [bs=1, obj_num=1, T, c, h, w] or [bs=1, obj_num=1, T, hw, c]

            # readout memory: query <- [query: ref memorys] in convmae.block3
            [f16, f8, f4] = self.network(    # [bs=1, obj_num=1, c, h, w]
                'mixattn_memory_query',
                query_frame=frame, ref_block_values=memory_value,
            )

            f16_list.append(f16)
            f8_list.append(f8)
            f4_list.append(f4)

        f16_list = torch.cat(f16_list, dim=1)   # [bs=1, obj_num, c, h, w]
        f8_list = torch.cat(f8_list, dim=1)   # [bs=1, obj_num, c, h, w]
        f4_list = torch.cat(f4_list, dim=1)   # [bs=1, obj_num, c, h, w]

        return [f16_list, f8_list, f4_list]

    def add_memory(self, frames_with_masks, memory_value:dict, objects: List[int], is_mem_frame):
        """

        @param frames_with_masks: [bs=1, all_obj_num, 3+2, H, W]
        @param memory_value: {
                f4, f8, f16: [bs=1, all_obj_num, c, h, w]
                block{index}: [bs=1, all_obj_num, hw, C]
            }
        @param objects: all object label list
        @param is_mem_frame:
        @return:
        """
        assert is_mem_frame or self.include_last
        self.memory_store.add(frames_with_masks=frames_with_masks, memory_value=memory_value, objects=objects, is_mem_frame=is_mem_frame)

    def create_cls_token(self, all_obj_num, init_cls_token):
        """

        @param obj_num: the TOTAL number of objects
        @param init_cls_token: [bs=1, N=1, C=768] from convmae.cls_token
        @return:
        """
        if self.cls_token is None:
            self.cls_token = init_cls_token.repeat(all_obj_num, 1, 1)    # [obj_num, N=1, C=768]
        elif self.cls_token.shape[0] != all_obj_num:
            init_cls_token = init_cls_token.repeat(all_obj_num - self.cls_token.shape[0], 1, 1)
            self.cls_token = torch.cat([self.cls_token, init_cls_token], dim=0)

        assert self.cls_token.shape[0] == all_obj_num

    def set_cls_token(self, update_cls_token):
        """

        @param update_cls_token: [obj_num, N=1, C=768]
        @return:
        """
        assert self.cls_token.shape == update_cls_token.shape
        self.cls_token = update_cls_token

    def get_cls_token(self, obj_num):
        """

        @param obj_num:
        @return:
        """
        assert self.cls_token.shape[0] == obj_num
        return self.cls_token