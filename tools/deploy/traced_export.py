# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import argparse
import io
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch

path = os.path.abspath('.')
sys.path.append(path)

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.file_io import PathManager
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger
import wtorch.utils as wtu
# import some modules added in project like this below
# sys.path.append("projects/FastDistill")
# from fastdistill import *

setup_logger(name="fastreid")
logger = logging.getLogger("fastreid.onnx_export")


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

mot17_config = {
    "config_file":"configs/MOT17/sbs_S50.yml",
    "ckpt":"weights/mot17_sbs_S50.pth",
}
mot20_config = {
    "config_file":"configs/MOT20/sbs_S50.yml",
    "ckpt":"weights/mot20_sbs_S50.pth",
}
mot20_config_t0 = {
    "config_file":"configs/MOT20/sbs_S50.yml",
    "ckpt":"/home/wj/ai/mldata1/MOT_output/weights/logs/sbs_S50/model_final.pth",
}
sportsmot_config = {
    "config_file":"configs/SportsMOT/sbs_S50.yml",
    "ckpt":"/home/wj/ai/mldata1/SportsMOT-2022-4-24/logs/sbs_S50/model_final.pth",
}
sportsmot_config_t0 = {
    "config_file":"configs/SportsMOT/sbs_S50.yml",
    "ckpt":"/home/wj/ai/mldata1/SportsMOT-2022-4-24/logs/sbs_S50_finetune/model_final.pth",
}
export_config = mot20_config_t0
def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to ONNX model")

    parser.add_argument(
        "--config-file",
        default=export_config["config_file"],
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--name",
        default="baseline",
        help="name for converted model"
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of onnx runtime"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def export_traced_model(model, inputs):
    """
    Trace and export a model to traced format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an traced model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, traced may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    logger.info("Beginning ONNX file converting")
    # Export the model to ONNX
    with torch.no_grad():
        traced_model = torch.jit.trace(
                model,
                inputs,
                )

    return traced_model


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if cfg.MODEL.HEADS.POOL_LAYER == 'FastGlobalAvgPool':
        cfg.MODEL.HEADS.POOL_LAYER = 'GlobalAvgPool'
    model = build_model(cfg)
    ckpt = export_config['ckpt']
    print(f"Load {ckpt}")
    checkpoint = torch.load(ckpt)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    wtu.forgiving_state_restore(model,checkpoint)
    if hasattr(model.backbone, 'deploy'):
        model.backbone.deploy(True)
    model.eval()
    logger.info(model)

    inputs = torch.randn(args.batch_size, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).to(model.device)
    print(f"Input size: {inputs.shape}")
    traced_model = export_traced_model(model, inputs)

    PathManager.mkdirs(args.output)

    save_path = os.path.join(args.output, args.name+'.torch')
    traced_model.save(save_path)
    logger.info("Traced model model file has already saved to {}!".format(save_path))
