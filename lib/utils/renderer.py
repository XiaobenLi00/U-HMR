import trimesh
import pyrender
import numpy as np
from torchvision.utils import make_grid
import torch
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0] \
    if 'GPU_DEVICE_ORDINAL' in os.environ.keys() else '0'

def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.img_res = img_res
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(
                vertices[i], camera_translation[i], images_np[i]), (2, 0, 1))).float()
            rend_img_side = torch.from_numpy(np.transpose(self.__call__(
                vertices[i], camera_translation[i], images_np[i], side_view = True), (2, 0, 1))).float()
            rend_img_top = torch.from_numpy(np.transpose(self.__call__(
                vertices[i], camera_translation[i], images_np[i], top_view = True), (2, 0, 1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(rend_img_top)
        rend_imgs = make_grid(rend_imgs, nrow=4)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, side_view=False, top_view = False, rot_angle=90):
        renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,viewport_height=self.img_res,point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.865, 0.915, 0.98, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        if top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        # light_pose = np.eye(4)

        # light_pose[:3, 3] = np.array([0, -1, 1])
        # scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([0, 1, 1])
        # scene.add(light, pose=light_pose)

        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        # output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)

        if side_view or top_view:
            output_img = color[:, :, :3]
        # if not side_view:
        else:
            output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image)
        # else:
        #     output_img = color[:, :, :3]
        renderer.delete()
        return output_img