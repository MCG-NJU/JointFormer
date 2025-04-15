from jointformer.inference.memory_manager import MemoryManager
from jointformer.model.network import JointFormer
from jointformer.model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network:JointFormer, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']

        self.include_last = config['include_last']
        self.max_memory_frames = config['max_memory_frames']

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0

        self.memory = MemoryManager(config=self.config, network=self.network)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        """
        segment current frame, update object if there has new objects, finally add memory if is_mem_frame or include_last
        the size of input and output are all arg.size
        :param image: [3, H, W]
        :param mask: None, or [num_obj, H, W] when first frame or use_all_mask
                    if this mask is not from first frame, it may only contain new objects' label
                    so we not only use it to fix mask prediction, but also need to copy old objects' mask from mask prediction to this mask GT
        :param valid_labels:
            if there is new objects, valid_labels is list(new_obj_num) if coherent, else range: new_obj_num which also can been seen as list
            if no new objects, None
        :param end: if this frame this the last frame
        :return: [num_obj+1, H, W]
        """

        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        """
            is_mem_frame: if True this frame will be used to update memory,
                set True if (reach memory interval, or mask GT is provided means first frame / new objects coming),
                and not last frame
            need_segment: if True this frame need to segment,
                set True if it is not the first frame
                and (no new label list, or the new label list is not from first frame (new label list is provided but there is already old object)) 
        """

        # segment the current frame is needed
        if need_segment:
            # 1) readout memory: query <- [query: ref memorys] in convmae.block3
            multi_scale_features = self.memory.match_memory(frame=image)
            # multi_scale_features(f16, f8, f4)[bs=1, obj_num, c, h, w]

            # 2) enhance f16 with cls_token: f16 <- cls_token in convmae.block5
            # note that cls_token was updated with last frame
            enhanced_f16 = self.network(  # [bs=1, obj_num, 768, h16, w16]
                'enhance_query',
                f16=multi_scale_features[0],
                cls_token=self.memory.get_cls_token(obj_num=len(self.memory.memory_store.all_objects)).unsqueeze(0)
            )

            # 3) decoder: use readout multi-scale memory and enhanced f16, output mask prediction and fusion_feature
            fusion_feature, _, pred_prob_with_bg = self.network(
                'segment',
                multi_scale_features=multi_scale_features, enhanced_f16=enhanced_f16,
                selector=None, strip_bg=False
            )  # fusion_feature: [bs=1, obj_num, c, h, w], pred_prob_with_bg: [bs=1, old_obj_num+1, H_pad, W_pad]

            # 4) update cls_token(1) with fusion_feature: cls_token <- [cls_token: fusion_feature] in convmae.block4
            cls_token = self.network(  # [bs, max_obj_num, N=1, C=768]
                'mixattn_feature_clstoken',
                fusion_feature=fusion_feature,
                cls_token=self.memory.get_cls_token(obj_num=len(self.memory.memory_store.all_objects)).unsqueeze(0)
            )

            pred_prob_with_bg = pred_prob_with_bg[0]    # remove batch dim, [old_obj_num+1, H_pad, W_pad]
            pred_prob_no_bg = pred_prob_with_bg[1:]     # [old_obj_num, H_pad, W_pad]

            # update(1) cls_token
            self.memory.set_cls_token(update_cls_token=cls_token.squeeze(0))
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)  # [obj_num, H_pad, W_pad]

            if pred_prob_no_bg is not None:
                assert mask.shape[0] == len(self.all_labels)
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)  # [1+obj_num, H_pad, W_pad]

            # also create cls_token for new objects.
            self.memory.create_cls_token(all_obj_num=len(self.all_labels), init_cls_token=self.network.get_cls_token())

        # save as memory if needed
        if (is_mem_frame or self.include_last) and not end:
            # 5) save this frame's mask prediction: this memory self-attn, and update(2) cls_token in convmae.block3
            frames_with_masks = self.network(  # [bs=1, all_obj_num, 3+2, H_pad, W_pad]
                'encode_value',
                frame=image, masks=pred_prob_with_bg[1:].unsqueeze(0)
            )
            block_values, cls_token = self.network( # [bs=1, all_obj_num, c, h, w] or [bs=1, all_obj_num, hw, C], [bs=1, all_obj_num, N=1, C]
                'mixattn_memory_clsToken',
                memory=frames_with_masks,  # [bs=1, all_obj_num, 3+2, H_pad, W_pad]
                cls_token=self.memory.get_cls_token(len(self.all_labels)).unsqueeze(0),  # [bs=1, all_obj_num, N=1, C=768]
                backbone_update_clsToken=True
            )

            # save in memory bank
            self.memory.add_memory(
                frames_with_masks=frames_with_masks, memory_value=block_values,
                objects=self.all_labels, is_mem_frame=is_mem_frame)

            if is_mem_frame:    # here
                self.last_mem_ti = self.curr_ti

            # update(2) cls_token: we update cls_token every frame
            self.memory.set_cls_token(update_cls_token=cls_token.squeeze(0))
                
        return unpad(pred_prob_with_bg, self.pad)
