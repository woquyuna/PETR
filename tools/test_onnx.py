# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

def voxelgrid_generator(img_metas, model_cfg):
    depth_num = 64
    depth_start = 1
    position_range = [-65, -65, -8.0, 65, 65, 8.0]
    LID = True
    if 'depth_num' in model_cfg['pts_bbox_head'].keys():
        depth_num = model_cfg['pts_bbox_head']['depth_num']
    if 'depth_start' in model_cfg['pts_bbox_head'].keys():
        depth_start = model_cfg['pts_bbox_head']['depth_start']
    if 'LID' in model_cfg['pts_bbox_head'].keys():
        LID = model_cfg['pts_bbox_head']['LID']
    if 'position_range' in model_cfg['pts_bbox_head'].keys():
        position_range = model_cfg['pts_bbox_head']['position_range']

    pad_h, pad_w, _ = img_metas['pad_shape'][0]
    H, W = pad_h // 32, pad_w // 32
    coords_h = torch.arange(H).float() * pad_h / H
    coords_w = torch.arange(W).float() * pad_w / W

    if LID:
        index = torch.arange(start=0, end=depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (position_range[3] - depth_start) / (depth_num * (1 + depth_num))
        coords_d = depth_start + bin_size * index * index_1
    else:
        index = torch.arange(start=0, end=depth_num, step=1).float()
        bin_size = (position_range[3] - depth_start) / depth_num
        coords_d = depth_start + bin_size * index

    eps = 1e-5
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    # coords[..., :2] = coords[..., :2] * self.maximum_alter(coords[..., 2:3], torch.ones_like(coords[..., 2:3]) * eps)

    return coords

def img2lidar_tensor_generator(img_metas):
    img2lidars = []
    for img_meta in img_metas:
        img2lidar = []
        for i in range(len(img_meta['lidar2img'])):
            img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
        img2lidars.append(np.asarray(img2lidar))
    img2lidars = np.asarray(img2lidars)
    img2lidars = torch.tensor(img2lidars, dtype=torch.float32)     # (B, N, 4, 4)
    # print("*" * 50)
    # print(img2lidars.shape)
    # print(img2lidars)
    return img2lidars



class nuscenceData:
    def __init__(self, data_loader):
        self._data_loader = data_loader

    def load_data(self, model_cfg):
        for data in self._data_loader:
            img = data['img'][0].data[0]
            B, N, C, H, W = img.size()
            data['img'][0] = img.reshape(B * N, C, H, W)

            img_metas = data['img_metas'][0].data[0][0]
            img_metas['lidar2img'] = img2lidar_tensor_generator([img_metas])
            img_metas['voxelgrid'] = voxelgrid_generator(img_metas, model_cfg)
            img_metas['sin_embed'] = np.fromfile("/mnt/apollo/input_sin_embed_1x6x256x15x25.bin", dtype=np.float32)
            img_metas['sin_embed'] = torch.tensor(img_metas['sin_embed'].reshape(1, 6, 256, 15, 25), dtype=torch.float32)
            img_metas['query_embeds'] = np.fromfile("/mnt/apollo/input_query_embeds_900x256.bin", dtype=np.float32)
            img_metas['query_embeds'] = torch.tensor(img_metas['query_embeds'].reshape(900, 256), dtype=torch.float32)

            data['img_metas'][0] = []
            data['img_metas'][0].append(img_metas)

            # save input
            # data['img'][0].cpu().detach().numpy().tofile('/mnt/apollo/img.bin')
            # img_metas['lidar2img'].cpu().detach().numpy().tofile('/mnt/apollo/lidar2img.bin')

            # print(img_metas)

            return data, data['img'][0], img_metas['lidar2img'],  img_metas['voxelgrid']

class PetrWrapper(torch.nn.Module):
    def __init__(self, org_model):
        super(PetrWrapper, self).__init__()
        self.org_model = org_model
        self.data = None

    def set_data(self, data):
        self.data = data

    def forward(self, img, lidar2img):
        with torch.no_grad():
            outs = self.org_model(return_loss=False, rescale=True,**self.data)
        all_cls_scores = outs['all_cls_scores'][-1]
        all_bbox_preds = outs['all_bbox_preds'][-1]
        print(all_cls_scores.shape)
        print(all_bbox_preds.shape)
        # all_cls_scores = all_cls_scores.reshape(3, 1, 900, 10)
        # all_bbox_preds = all_bbox_preds.reshape(3, 1, 900, 10)
        all_cls_scores.cpu().detach().numpy().tofile('/mnt/apollo/all_cls_scores.bin')
        all_bbox_preds.cpu().detach().numpy().tofile('/mnt/apollo/all_bbox_preds.bin')

        return all_cls_scores, all_bbox_preds

def get_onnx_model(model, img, lidar2img, grid, out_path, device):
    model.eval()
    model = model.to(device)
    img_tensor = img.to(device)
    lidar2img = lidar2img.to(device)
    grid = grid.to(device)

    torch.onnx.export(
        model,
        tuple([img_tensor, lidar2img]),
        # tuple([img_tensor, lidar2img, grid]),
        out_path,
        export_params=True,
        opset_version=11,
        input_names=["img", "lidar2img"],
        # input_names=["img", "lidar2img", "grid"],
        output_names=["all_cls_scores", "all_bbox_preds"],
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        verbose=True
    )
    print("export onnx success:", out_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # print(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    dataset = data_loader.dataset
    for data in dataset:
        model.set_metas(data['img_metas'][0].data)
        break

    nus_data = nuscenceData(data_loader)
    data, img, lidar2img, grid = nus_data.load_data(cfg.model)
    petr_net = PetrWrapper(model)
    petr_net.set_data(data)
    get_onnx_model(petr_net, img, lidar2img, grid, "/mnt/apollo/hozonlinearc_epoch32.onnx", 'cpu')

    import sys
    sys.exit(0)


    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
