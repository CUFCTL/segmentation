import contextlib

import numpy as np
import torch
import torch.onnx
from tqdm import tqdm
from time import perf_counter


import lib.cylib as cylib

__all__ = ['compute_errors', 'get_pred', 'evaluate_semseg', 'evaluate_semseg_timing', 'evaluate_semseg_live_video']


def compute_errors(conf_mat, class_info, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(1)
    TPFN = conf_mat.sum(0)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def get_pred(logits, labels, conf_mat):
    _, pred = torch.max(logits.data, dim=1)
    pred = pred.byte().cpu()
    pred = pred.numpy().astype(np.int32)
    true = labels.numpy().astype(np.int32)
    cylib.collect_confusion_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)


def mt(sync=False):
    if sync:
        torch.cuda.synchronize()
    return 1000 * perf_counter()


def evaluate_semseg(model, data_loader, class_info, observers=()):
    model.eval()
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['original_labels'].shape[1:3])
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            # The observer objected is already created so when it is called the parameters get passed into its __call__ function
            for o in observers: 
                o(pred, batch, additional)
            cylib.collect_confusion_matrix(pred.flatten(), batch['original_labels'].flatten(), conf_mat)
        print('')
        pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=True)
    model.train()
    return iou_acc, per_class_iou


def evaluate_semseg_timing(model, data_loader, class_info, observers=(), eval_per_steps=20):
    """Runs inference on the model and calculates the time. Currently, this function
    does not calucate metrics, does not apply a color map, and does not save the segmented image
    as this slows down inference speed

    Args:
        eval_per_steps: Evaluate the fps every certain number of steps
    """
    model.eval()
    n = len(data_loader)
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        torch.cuda.synchronize()  # Wait for all operations to finish so we can begin timing (cuda operations are asynchronous)
        start_t = perf_counter()  
        for step, batch in tqdm(enumerate(data_loader), total=n, disable=True):
            #batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['image'].shape[2:4]) # shape[2:4] grabs (H, W)
            # batch['original_labels'] does not exist during live inference so I changed thie to batch['image']
            #pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            # Trying to time exactly like SwiftNet
            # All of the max value methods below have similar timings
            # (Keeping the one swift net uses)
            #pred = torch.argmax(logits.data, dim=1).byte().cpu()
            #pred = logits.data.argmax(dim=1).byte().cpu()
            # Not transfering the output back to the cpu speeds up the fps by about 5
            #pred = torch.argmax(logits.data, dim=1)

            _, pred = logits.max(dim=1)
            out = pred.data.byte().cpu()
            if eval_per_steps == 1: 
                torch.cuda.synchronize()
                curr_t = perf_counter()
                print(f'{((step+1) * 1) / (curr_t - start_t):.2f}fps')
            elif step % eval_per_steps == 0 and step > 0: # Check fps every 'eval_per_steps'
                torch.cuda.synchronize()  # Wait for all operations to finish that are being timed
                curr_t = perf_counter()
                # Do not need to reset start_t=0 bc we are dividing into the total number of frames processed
                # step+1 because the first step size will have 1 extra image
                # then every step after the first will have the correct amount
                print(f'{((step+1)* 1) / (curr_t - start_t):.2f}fps') # The '* 1' represents batch_size I think
                # step is the group of each batch size. Ex: 500 test images and batch_size=5 => 100 steps
        
    return


def evaluate_semseg_live_video(model, data_loader, class_info, observers=(), eval_per_steps=20):
    """

    Args:
        eval_per_steps: Evaluate the fps every certain number of steps
    """
    model.eval()
    managers = [torch.no_grad()] + list(observers)
    with contextlib.ExitStack() as stack:
        for ctx_mgr in managers:
            stack.enter_context(ctx_mgr)
        conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
        torch.cuda.synchronize()  # Wait for all operations to finish so we can begin timing (cuda operations are asynchronous)
        start_t = perf_counter()  
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
            #batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
            logits, additional = model.do_forward(batch, batch['image'].shape[2:4]) # shape[2:4] grabs (H, W)
            # batch['original_labels'] does not exist during live inference so I changed this to batch['image']
            pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)
            
            if eval_per_steps == 1:
                torch.cuda.synchronize()
                curr_t = perf_counter()
                print(f'{(1 * 1) / (curr_t - start_t):.2f}fps')
            elif step % eval_per_steps == 0 and step > 0:
                torch.cuda.synchronize()
                curr_t = perf_counter()
                print(f'{(step * 1) / (curr_t - start_t):.2f}fps')
            for o in observers:
                colored_pred = o(pred)
    return colored_pred
