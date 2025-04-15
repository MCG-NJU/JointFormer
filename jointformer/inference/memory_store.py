import torch
from typing import List

class MemoryStore:

    def __init__(self, config):
        self.config = config
        self.include_last = config['include_last']
        self.max_memory_frames = config['max_memory_frames']

        self.all_objects = []   # containing old objects, extend when objects List has new objects.

        self.mem_frames_with_masks = dict() # [obj] = [T, 3+2, H, W]
        if self.include_last:
            self.tmp_mem_frames_with_masks = dict() # [obj] = [T=1, 3+2, H, W]

        self.memory_value = dict()  # [obj] = [T, c, h, w] or [T, hw, c]
        if self.include_last:
            self.tmp_memory_value = dict()  # [obj] = [T=1, c, h, w] or [T=1, hw, c]

    def add(self, frames_with_masks, memory_value:dict, objects: List[int], is_mem_frame):
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
        frames_with_masks = frames_with_masks[0]    # remove batch_size [all_obj_num, 3+2, H, W]
        assert frames_with_masks.shape[0] == len(objects)

        all_obj_memory_value = []  # list(all_obj_num): dict()
        for obj in range(len(objects)):
            obj_memory_value = dict()
            for k in memory_value.keys():
                value = memory_value[k] # [bs=1, all_obj_num, c, h, w] or [bs=1, all_obj_num, hw, C]
                assert value.shape[0] == 1 and value.shape[1] == len(objects)
                obj_memory_value[k] = value[0, obj].unsqueeze(0)  # [T=1, c, h, w] or [T=1, hw, c]
            assert obj_memory_value.keys() == memory_value.keys()
            all_obj_memory_value.append(obj_memory_value)

        if objects is not None:
            # First consume objects that are already in the memory bank
            # cannot use set here because we need to preserve order
            # shift by one as background is not part of value
            remaining_objects = [obj - 1 for obj in objects]

            # (1) For consume objects that are already in the memory bank, we need to put value into old object list
            for old_obj in self.all_objects:
                assert old_obj in remaining_objects
                remaining_objects.remove(old_obj)

                if is_mem_frame:    # [T+1, 3+2, H, W]
                    if self.max_memory_frames == 1: # only save the first frame with gt mask for each object.
                        assert (self.mem_frames_with_masks[old_obj].shape[0] == 1)

                        # replace the last_frame if max_memory_frames == 1
                        self.tmp_mem_frames_with_masks[old_obj] = frames_with_masks[old_obj].unsqueeze(0)
                        self.tmp_memory_value[old_obj] = all_obj_memory_value[old_obj]
                    else:
                        if (self.max_memory_frames > 0) and (self.mem_frames_with_masks[old_obj].shape[0] == self.max_memory_frames):
                            self.remove_mem_frames_with_masks(old_obj=old_obj, idx=1)
                            assert (self.mem_frames_with_masks[old_obj].shape[0] == (self.max_memory_frames - 1))
                            self.remove_memory_value(old_obj=old_obj, idx=1)

                        self.mem_frames_with_masks[old_obj] = torch.cat(
                            [self.mem_frames_with_masks[old_obj], frames_with_masks[old_obj].unsqueeze(0)], 0)
                        self.append_memory_value(old_obj=old_obj, obj_memory_value=all_obj_memory_value[old_obj])

                elif self.include_last: # [T=1, 3+2, H, W]
                    self.tmp_mem_frames_with_masks[old_obj] = frames_with_masks[old_obj].unsqueeze(0)
                    self.tmp_memory_value[old_obj] = all_obj_memory_value[old_obj]
                else:
                    raise 'must is_mem_frame or include_last'

            # (2) If there are remaining objects, means these are new objects
            if len(remaining_objects) > 0:
                assert (is_mem_frame is True), 'if a frame has new objects, it must be is_mem_frame'

                for new_obj in remaining_objects:
                    self.mem_frames_with_masks[new_obj] = frames_with_masks[new_obj].unsqueeze(0)   # [T=1, 3+2, H, W]
                    self.memory_value[new_obj] = all_obj_memory_value[new_obj]

                self.all_objects.extend(remaining_objects)  # add new objects into memory object list.
                assert sorted(self.all_objects) == self.all_objects, 'Objects MUST be inserted in sorted order.'

        else:
            raise NotImplementedError

    def remove_mem_frames_with_masks(self, old_obj, idx):
        assert idx != 0

        assert self.mem_frames_with_masks[old_obj].shape[0] > idx
        self.mem_frames_with_masks[old_obj] = torch.cat([  # here
            self.mem_frames_with_masks[old_obj][0:idx], self.mem_frames_with_masks[old_obj][(idx + 1):]
        ], dim=0)

        return

    def remove_memory_value(self, old_obj, idx):
        assert idx != 0

        memory_value = self.memory_value[old_obj]

        new_memory_value = dict()
        for k in memory_value.keys():
            value = memory_value[k]  # [T, c, h, w] or [T, hw, c]
            assert value.shape[0] > idx
            value = torch.cat([value[0:idx], value[idx + 1:]], dim=0)
            assert value.shape[0] == self.max_memory_frames - 1
            new_memory_value[k] = value

        self.memory_value[old_obj] = new_memory_value

    def append_memory_value(self, old_obj, obj_memory_value):

        memory_value = self.memory_value[old_obj]
        assert memory_value.keys() == obj_memory_value.keys()

        new_memory_value = dict()
        for k in memory_value.keys():
            value = memory_value[k] # [T, c, h, w] or [T, hw, C]
            obj_value = obj_memory_value[k] # [T=1, c, h, w] or [T=1, hw, C]
            assert value.shape[1:] == obj_value.shape[1:]

            new_memory_value[k] = torch.cat([value, obj_value], dim=0)

            assert new_memory_value[k].shape[0] <= self.max_memory_frames and \
                   new_memory_value[k].shape[0] == self.mem_frames_with_masks[old_obj].shape[0]

        self.memory_value[old_obj] = new_memory_value