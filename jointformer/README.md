## convmae.py

### inference_wrong (eval_wrong.py):
    
- "mem_every" will not be arrived since we update self.last_mem_ti = self.curr_ti every step in error,
- so "is_mem_frame" is always false (unless new object coming with its new mask gt), we replace the new frame with last_frame.

### inference (eval_wrong.py):

- if "max_memory_frames" > 1
  - when arrived "mem_every"(or new object coming with its new mask gt), we save current frame in memory bank, else replace the last_frame. 
  - Before saving in memory bank, we check the memory size: if memory size == "max_memory_frames", we just "drop the 1-th frame" (MUST KEEP THE 0-th FRAME!)
- if "max_memory_frames" == 1
  - we keep the 0-th frame for each object unchanged, and replace the last_frame in each step whatever the "is_mem_frame" is

## convmae_topK.py

**do top_k in some blocks (topk_num, topk_range, topk_block)**

### inference_topK (eval_topK.py): topk_num, topk_range, topk_block

- if "max_memory_frames" > 1
  - when arrived "mem_every"(or new object coming with its new mask gt), we save current frame in memory bank, else replace the last_frame. 
  - Before saving in memory bank, we check the memory size: if memory size == "max_memory_frames", we just "drop the 1-th frame" (MUST KEEP THE 0-th FRAME!)
- if "max_memory_frames" == 1
  - we keep the 0-th frame for each object unchanged, and replace the last_frame in each step whatever the "is_mem_frame" is

## comvmae_usage.py: return_usage


**return usage [obj, T] for each object/frame in the last block**


### inference_usage_onlyshort (eval_usage_onlyshort.py): return_usage
- Update usage:
  - In each step, while predicting the current frame mask, we also get the usage for memory frame, not contain last_frame
  - Whatever the "is_mem_frame" is, we update the usage and life before adding/replace.
- if "max_memory_frames" > 1
  - When arrived "mem_every"(or new object coming with its new mask gt), we save current frame in memory bank, else replace the last_frame. 
  - Before saving in memory bank, we check the memory size
    - if memory size == "max_memory_frames", we drop the useless memory frame (MUST KEEP THE 0-th FRAME!)
    - useless means the minimum norm_usage(usage / life)
- if "max_memory_frames" == 1
  - we keep the 0-th frame for each object unchanged, and replace the last_frame in each step whatever the "is_mem_frame" is


### inference_usage_short_long (eval_usage_short_long.py): return_usage

- Update usage:
  - In each step, while predicting the current frame mask, we also get the short-usage list(not contain last_frame) and long-usage list
  - the usage is list(short-term object), so long-usage list will may have Tensor(0,)
  - Whatever the "is_mem_frame" is, we update the usage and life before adding/replace.
- if "max_memory_frames" > 1
  - When arrived "mem_every"(or new object coming with its new mask gt), we save current frame in memory bank, else replace the last_frame. 
    - If is_memory == Ture, we append current frame into short-term memory directly
    - Then check if this object size more than max_short_term
      - If more, compress long_term if long_term is also full, then select prototype from short-term (MUST KEEP THE 0-th FRAME!), and move them into long-term
  - useless means the minimum norm_usage(usage / life)
- if "max_memory_frames" == 1
  - we keep the 0-th frame for each object unchanged, and replace the last_frame in each step whatever the "is_mem_frame" is
  - only use w/o long-term memory

