import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from .HRNet_Semantic_Segmentation_light_weight.lib.models.seg_hrnet import HighResolutionNet
from .HRNet_Semantic_Segmentation_light_weight.lib.config import config
from .HRNet_Semantic_Segmentation_light_weight.lib.config import update_config
from .HRNet_Semantic_Segmentation_light_weight.lib.utils.utils import FullModel


def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def aux_net():
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpu = [0]

    # hard coding config file for now
    #cfg_file = 'HRNet_Semantic_Segmentation_light_weight/experiments/cityscapes/hrnet_w18_small_v1_auxnet.yaml'
    cfg_file = 'aux_adapt/HRNet_Semantic_Segmentation_light_weight/experiments/cityscapes/hrnet_w18_small_v1_auxnet.yaml'

    # Uses yacs global config method rather than local method
    update_config(config, cfg_path=cfg_file)

    # build model
    model = get_seg_model(config)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    #train_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                    root=config.DATASET.ROOT,
    #                    list_path=config.DATASET.TRAIN_SET,
    #                    num_samples=None,
    #                    num_classes=config.DATASET.NUM_CLASSES,
    #                    multi_scale=config.TRAIN.MULTI_SCALE,
    #                    flip=config.TRAIN.FLIP,
    #                    ignore_label=config.TRAIN.IGNORE_LABEL,
    #                    base_size=config.TRAIN.BASE_SIZE,
    #                    crop_size=crop_size,
    #                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
    #                    scale_factor=config.TRAIN.SCALE_FACTOR)
#
    #trainloader = torch.utils.data.DataLoader(
    #    train_dataset,
    #    batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
    #    shuffle=config.TRAIN.SHUFFLE,
    #    num_workers=config.WORKERS,
    #    pin_memory=True,
    #    drop_last=True)

    class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=config.TRAIN.IGNORE_LABEL,
                                 weight=class_weights)

    model = model
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    
    return model, optimizer, criterion
    





