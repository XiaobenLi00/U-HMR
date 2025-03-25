# export PYOPENGL_PLATFORM=osmesa
# python infer.py --cfg_name experiments/h36m/resnet50_pos_3view.yaml --image_dir ./test_data
import torch
from lib.utils.config import get_config
from lib.models.fusion import Mv_Fusion

from lib.utils.renderer import Renderer
from lib.models.smpl_wrapper import SMPL
from lib.utils import vis
import os
import time
import argparse
import random
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from einops import rearrange
import trimesh


def process_images(image_folder, cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*')))
    inputs = []
    
    for path in image_paths[:4]:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()  # [1, C, H, W]
        inputs.append(img_tensor)
        
    return inputs

def convert_to_image(keypoints_2d_vis,images_pred, save_dir):
    if keypoints_2d_vis.dtype == np.float32:
        array = (array * 255).astype(np.uint8)
    elif keypoints_2d_vis.dtype != np.uint8:
        raise ValueError("array must be uint8 or floa32")
    
    if keypoints_2d_vis.shape[0] == 4:  # RGBA
        keypoints_2d_vis = keypoints_2d_vis.transpose(1, 2, 0)
    Image.fromarray(keypoints_2d_vis, 'RGBA').save(os.path.join(save_dir, f'keypoints_2d_vis1.png'))


    images_pred = images_pred.detach().cpu()
    if images_pred.dim() == 3 and images_pred.size(0) == 3:
        images_pred = images_pred.permute(1, 2, 0)  # H x W x 3

    if images_pred.dtype == torch.float32:
        images_pred = (images_pred * 255).clamp(0, 255).to(torch.uint8)
    Image.fromarray(images_pred.numpy(), 'RGB').save(os.path.join(save_dir, f'mesh_vis1.png'))





def visualize_results(input,output, n_views, mesh_renderer, smpl):
    images = torch.cat(input, dim=0)
    images = rearrange(images, "(n b) c d e -> (b n) c d e", n=n_views)
  
    pred_keypoints_2d = rearrange(output['pred_keypoints_2d'], "(n b) c d -> (b n) c d", n=n_views)

    keypoints_2d_vis = vis.visualize_2d_pose(images, pred_keypoints_2d)


    images = images.detach() * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
    

    pred_vertice = rearrange(output['pred_vertices'], "(n b) c d -> (b n) c d", n=n_views)
    pred_cam_t = rearrange(output['pred_cam_t'], "(n b) c -> (b n) c", n=n_views)
    
    images_pred = mesh_renderer.visualize_tb(pred_vertice.detach(), pred_cam_t.detach(), images)

    convert_to_image(keypoints_2d_vis, images_pred,"./vis")
    
    mesh_vertices = rearrange(pred_vertice, "(b n) c d -> b n c d", n=n_views)
    for b in range(mesh_vertices.shape[0]):
        for n in range(mesh_vertices.shape[1]):
            mesh_vertice = mesh_vertices.clone().detach().cpu().numpy()[b, n]
            vertex_colors = np.ones([mesh_vertice.shape[0], 4]) * [0.82, 0.9, 0.98, 1.0]
            face_colors = np.ones([smpl.faces.shape[0], 4]) * [0.82, 0.9, 0.98, 1.0]
            mesh = trimesh.Trimesh(mesh_vertice, smpl.faces, face_colors=face_colors, vertex_colors=vertex_colors, process=False)
            mesh.export(f'mesh_{b}_{n}.obj')


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg_name', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--image_dir', help='input your test image dir', required=True, type=str)
    args = parser.parse_args()
    cfg = get_config(args.cfg_name, merge=False)


    gpus=[0]
    model = Mv_Fusion(cfg, tensorboard_log_dir=None)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()


    smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
    smpl = SMPL(**smpl_cfg)
    
    mesh_renderer = Renderer(focal_length=cfg.EXTRA.FOCAL_LENGTH,
                        img_res=256, faces=smpl.faces)

    checkpoint = torch.load(cfg.TEST.MODEL_FILE)
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    input_sub = process_images(args.image_dir, cfg)
    n_views = len(input_sub)

    with torch.no_grad():
        output = model.module.forward_step(
            input_sub,
            n_views
        )

    visualize_results(input_sub,output,n_views,mesh_renderer,smpl)
if __name__ == "__main__":
    infer()