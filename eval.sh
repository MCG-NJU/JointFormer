export PYTHONPATH=$PYTHONPATH:./

topk_range="all_frames"
topk_block="all_blocks"
topk_clstoken="N"
output="./output"

### DAVIS 2017 val

dataset="D17"
split="val"
max_memory_frames="1"
topk_num="60"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### DAVIS 2017 test

dataset="D17"
split="val"
max_memory_frames="1"
topk_num="60"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### Youtube-VOS 2018

dataset="Y18"
split="val"
max_memory_frames="3"
topk_num="120"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}

### Youtube-VOS 2019

dataset="Y19"
split="val"
max_memory_frames="3"
topk_num="120"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### VOST

dataset="VOST"
split="val"
max_memory_frames="3"
topk_num="30"

model="./checkpoints/jointformer_s24.pth"
# model="./checkpoints/jointformer_s024.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### MOSE

dataset="MOSE"
split="val"
max_memory_frames="3"
topk_num="180"

model="./checkpoints/jointformer_s5.pth"
# model="./checkpoints/jointformer_s05.pth"
# model="./checkpoints/jointformer_s015.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### LVOS val

dataset="LVOS"
split="val"
max_memory_frames="3"
topk_num="120"

model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"
# model="./checkpoints/jointformer_s026.pth"
# model="./checkpoints/jointformer_s0126.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}

### LVOS test

dataset="LVOS"
split="test"
max_memory_frames="3"
topk_num="120"

model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"
# model="./checkpoints/jointformer_s026.pth"
# model="./checkpoints/jointformer_s0126.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}


### VISOR

dataset="VISOR"
split="val"
max_memory_frames="1"
topk_num="600"

model="./checkpoints/jointformer_s7.pth"
# model="./checkpoints/jointformer_s07.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}

### BURST val

dataset="BURST"
split="val"
max_memory_frames="3"
topk_num="30"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}

### BURST test

dataset="BURST"
split="test"
max_memory_frames="3"
topk_num="30"

model="./checkpoints/jointformer_s2.pth"
# model="./checkpoints/jointformer_s02.pth"
# model="./checkpoints/jointformer_s012.pth"

python jointformer/eval.py \
--model ${model} \
--output ${output} \
--dataset ${dataset} \
--split ${split} \
--topk_range ${topk_range} --topk_block ${topk_block} --topk_clstoken ${topk_clstoken} \
--max_memory_frames ${max_memory_frames} --topk_num ${topk_num}
