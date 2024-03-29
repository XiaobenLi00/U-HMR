import torch
from lib import datasets
from lib.utils.config import get_config
from lib.models.fusion import Mv_Fusion
from lib.utils.log_utils import create_logger, load_checkpoint, save_checkpoint
import os
import time
import argparse
import random
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg_name', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--dataset', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    cfg = get_config('experiments/{}/{}'.format(args.dataset, args.cfg_name), merge=False)
    cfg_name = args.cfg_name


    # cfg_name = 'vit_pos_trans_encoder.yaml'
    # cfg = get_config('experiments/h36m/{}'.format(cfg_name), merge= False)
    
    if cfg.IS_TRAIN:
        phase = 'train'
    else:
        phase = 'test'
    logger, final_output_dir, tensorboard_log_dir = create_logger(cfg, cfg_name, phase)
    gpus=[0]
    model = Mv_Fusion(cfg, tensorboard_log_dir)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # ckpt = torch.load(cfg.MODEL.PRETRAINED)['state_dict']
    # del_key = []
    # for key, _ in ckpt.items():
    #     if 'smpl_head.init_body_pose' == key:
    #         del_key.append(key)
    #     elif 'smpl_head.decpose.weight' == key:
    #         del_key.append(key)
    #     elif 'smpl_head.decpose.bias' == key:
    #         del_key.append(key)
    # for key in del_key:
    #     del ckpt[key]

    # model.module.load_state_dict(ckpt, strict=False)
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #     time.sleep(0.5)
    mocap_dataset = datasets.mocap_dataset(cfg.DATASET.MOCAP)
    train_dataset = eval('datasets.' + cfg.DATASET.TRAIN_DATASET)(cfg, cfg.DATASET.TRAIN_SUBSET, True)
    # train_dataset = MultiViewH36M(cfg, cfg.DATASET.TRAIN_SUBSET, True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=True)
    mocap_loader = torch.utils.data.DataLoader(
        mocap_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE * cfg.DATASET.N_VIEWS,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True)
    val_dataset = eval('datasets.' + cfg.DATASET.TEST_DATASET)(cfg, cfg.DATASET.TEST_SUBSET, False)
    # val_dataset = MultiViewH36M(cfg, cfg.DATASET.TEST_SUBSET, False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        drop_last = True,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=True)
    logger.info(f'=> Loaded datasets')
    len_train_data = len(train_loader)
    len_val_data = len(val_loader)
    best_perf = 1000000.0
    best_model = False
    if not cfg.IS_TRAIN:
        meters = {k: AverageMeter() for k in ['train_loss', 'val_loss', 'train_mpjpe', 'val_mpjpe', 'train_rec_error', 'val_rec_error']}
        model.eval()
        with torch.no_grad():
            model.module.load_state_dict(torch.load(cfg.TEST.MODEL_FILE)['state_dict'])
            # for i, (input, meta) in enumerate(val_loader):
            for i, data in enumerate(zip(val_loader, mocap_loader)):
                n_views = 4
                subset = random.sample(range(0, 4), n_views)
                subset.sort()
                (input, meta), mocap = data
                input_sub = []
                meta_sub = []
                for j in subset:
                    input_sub.append(input[j])
                    meta_sub.append(meta[j])
                model(input_sub, meta_sub, i, mocap, meters, len_val_data, n_views, train = False)
        # perf_indicator = meters['val_mpjpe'].avg
        logger.info(f'val_mpjpe: {meters["val_mpjpe"].avg}\t val_rec_error: {meters["val_rec_error"].avg}')
        return
    if cfg.TRAIN.RESUME:
        start_epoch, model = load_checkpoint(model, final_output_dir)
    for epoch in range(start_epoch, cfg.TRAIN.TOTAL_EPOCHS):
        meters = {k: AverageMeter() for k in ['train_loss', 'val_loss', 'train_mpjpe', 'val_mpjpe', 'train_rec_error', 'val_rec_error']}
        model.train()


        for i, data in enumerate(zip(train_loader, mocap_loader)):
            
            # n_views = random.sample(range(1,5),1)[0]
            n_views = 4

            # subset = random.sample(range(0, 4), cfg.DATASET.N_VIEWS)
            subset = random.sample(range(0, 4), n_views)
            subset.sort()
            (input, meta), mocap = data
            input_sub = []
            meta_sub = []
            for j in subset:
                input_sub.append(input[j])
                meta_sub.append(meta[j])    
            mocap_sub = {}
            for k,v in mocap.items():
                mocap_sub[k] = v[:cfg.TRAIN.BATCH_SIZE * n_views]
            model(input_sub, meta_sub, i, mocap_sub, meters, len_train_data, n_views, epoch, True)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(zip(val_loader, mocap_loader)):
                # subset = random.sample(range(0, 4), cfg.DATASET.N_VIEWS)
                n_views = 4
                subset = random.sample(range(0, 4), n_views)
                subset.sort()
                (input, meta), mocap = data
                input_sub = []
                meta_sub = []
                for j in subset:
                    input_sub.append(input[j])
                    meta_sub.append(meta[j])
                mocap_sub = {}
                for k,v in mocap.items():
                    mocap_sub[k] = v[:cfg.TRAIN.BATCH_SIZE * n_views]
                model(input_sub, meta_sub, i, mocap_sub, meters, len_val_data, n_views, epoch, False)
        logger.info(f'val_mpjpe: {meters["val_mpjpe"].avg}\t val_rec_error: {meters["val_rec_error"].avg}')
        perf_indicator = meters['val_mpjpe'].avg
        if perf_indicator < best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': model.module.optimizer.state_dict(),
        }, best_model, final_output_dir)
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    return

            


if __name__ == "__main__":
    main()
    exit()
    # from ...utils.config import get_config
    cfg_name = 'resnet50.yaml'
    from lib.models.heads.smpl_head import build_smpl_head, SMPLTransformerDecoderTokenHead, SMPLFCNFusionHead
    cfg = get_config('experiments/h36m/{}'.format(cfg_name), merge= False)
    # head = build_smpl_head(cfg)
    head = SMPLFCNFusionHead(cfg)
    features = torch.randn(4, 2048, 8, 8)
    pred_body_pose, pred_betas, pred_global_orientation, pred_cam = head(features)
    print(pred_body_pose.shape)
    print(pred_betas.shape)
    print(pred_global_orientation.shape)
    print(pred_cam.shape)