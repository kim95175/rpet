# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .rpet import build_rpet

def build_model(args):
    return build_rpet(args)
    
