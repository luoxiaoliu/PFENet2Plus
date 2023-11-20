import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.PFENetPlus import PFENet
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        default='/home/deep2/xiaoliu/PFENet++/config/pascal/pascal_split0_resnet50.yaml',
                        help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PFENet(layers=args.layers, classes=2, zoom_factor=8, \
                   criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=BatchNorm, \
                   pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg,args=argss)

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                               data_list=args.val_list, transform=val_transform, mode='val', \
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion)


##########################################################
def get_org_img(img):
    # print('img', img.shape)#3, 473, 473
    img = np.transpose(img, (1, 2, 0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img * std_vals + mean_vals
    img = img * 255
    # print('img', img.shape)
    return img


label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
from PIL import Image


def decode_labels(mask, num):
    """Decode batch of segmentation masks.

    Args:
      label_batch: result of inference after taking argmax.

    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < 21:
                # print(k)
                # print(label_colours[k])
                pixels[k_, j_] = label_colours[k * num]
            if k == 255:
                pixels[k_, j_] = (255, 255, 255)
    return np.array(img)  # (333, 500, 3)


def mask_to_img(mask, img, num):  # mask(333, 500)img(333, 500, 3)
    # print(mask.shape)
    # print(img.shape)
    mask_decoded = decode_labels(mask, num)  # 将mask转化成彩色图像

    # heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.addWeighted(img.astype(np.uint8), 0.8, mask_decoded.astype(np.uint8), 0.9, 0)  # 将原图和mask加权融合
    return img


###################################################
def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 1000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    ##############################################################
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.Resize(473),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    ##############################################################
    for e in range(20):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1
            # print(input.shape)#[1, 3, 473, 473]
            # print(s_input.shape)#[1, 1, 3, 473, 473]
            # print(target.shape)#[1, 473, 473]
            # print(s_mask.shape)#[1, 1, 473, 473]
            # if iter_num==86 or iter_num==129 or iter_num==175 or iter_num==194 or iter_num==196 or iter_num==205\
            #         or iter_num==295 or iter_num==366 or iter_num==376:
            #     print(s_path)
            #     print(q_path)
            # if iter_num==366:
            #    pred1 = torch.where(s_mask == 1, torch.full_like(s_mask, 255), torch.full_like(s_mask, 0))
            #    pred1 = pred1.squeeze().data.cpu().numpy().astype(np.int32)
            #
            #    cv2.imwrite('pred_%d.png' % (iter_num), pred1)
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            # print(input.shape)#[1, 3, 473, 473]
            target = target.cuda(non_blocking=True)
            # print(target.shape)#[1,473, 473]
            target_orign = target.cuda(non_blocking=True)
            # print(target_orign.shape)#[1,473, 473]
            ori_label = ori_label.cuda(non_blocking=True)
            # print(ori_label.shape)#[1, 366, 500]
            #######################
            # image_q = cv2.imread('/home/prlab/willow/Oneshot/PFENet-master/images/2008_000919.jpg', cv2.IMREAD_COLOR)
            # image_q = cv2.cvtColor(image_q, cv2.COLOR_BGR2RGB)
            # image_q = np.float32(image_q)
            # label_q = cv2.imread('/home/prlab/willow/Oneshot/PFENet-master/images/2008_000919.png',
            #                      cv2.IMREAD_GRAYSCALE)
            # image_s = cv2.imread('/home/prlab/willow/Oneshot/PFENet-master/images/2011_002189.jpg', cv2.IMREAD_COLOR)
            # image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
            # image_s = np.float32(image_s)
            # label_s = cv2.imread('/home/prlab/willow/Oneshot/PFENet-master/images/2011_002189.png',
            #                      cv2.IMREAD_GRAYSCALE)
            # image_q, label_q = val_transform(image_q, label_q)
            # image_s, label_s = val_transform(image_s, label_s)
            #
            # ###########################
            # input = image_q.unsqueeze(0).cuda(non_blocking=True)
            # target = label_q.unsqueeze(0).cuda(non_blocking=True)
            # target_orign = target.cuda(non_blocking=True)
            # ori_label = ori_label.cuda(non_blocking=True)
            # s_input = image_s.unsqueeze(0).unsqueeze(1).cuda(non_blocking=True)
            # s_mask = label_s.unsqueeze(0).unsqueeze(1).cuda(non_blocking=True)
            # target = torch.where(target == 1, torch.full_like(target, 1), torch.full_like(target, 0))
            # print(input.shape)
            # print(target.shape)
            # print(s_input.shape)
            # print(s_mask.shape)
            ################################################
            start_time = time.time()
            output1 = model(s_x=s_input, s_y=s_mask, x=input, y=target)  # [1, 2, 473, 473]
            # print(output1.shape)#[1, 2, 473, 473]
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output1, size=target.size()[1:], mode='bilinear', align_corners=True)
            # print(output.shape)
            # print(target.shape)
            loss = criterion(output, target)

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
            class_union_meter[(subcls - 1) % split_gap] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num / 100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # print(iou_class)
    # print(intersection_meter.sum)
    # print(union_meter.sum)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # print(target_meter.sum)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
