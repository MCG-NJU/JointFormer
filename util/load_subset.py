"""
load_subset.py - Presents a subset of data
DAVIS - only the training set
YouTubeVOS - I manually filtered some erroneous ones out but I haven't checked all
"""
import json


def load_sub(path='util/subsets/davis_subset.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset

def load_sub_empty(path='util/subsets/davis_empty_masks.txt'):
    with open(path, mode='r') as f:
        empty_masks = json.load(f)
    return empty_masks