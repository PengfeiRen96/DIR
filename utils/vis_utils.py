import pickle
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.manolayer import ManoLayer
from utils.utils import projection_batch, get_mano_path, get_dense_color_path

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    HardFlatShader,
    HardGouraudShader,
    AmbientLights,
    SoftSilhouetteShader
)
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)


class myShader(torch.nn.Module):

    def __init__(
            self, device="cpu", cameras=None, blend_params=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)

        return images


class MeshRendererWithFragments(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer (a MeshRasterizer or a MeshRasterizerOpenGL)
    and shader class which each have a forward function.
    In the forward pass this class returns the `fragments` from which intermediate
    values such as the depth map can be easily extracted e.g.
    .. code-block:: python
        images, fragments = renderer(meshes)
        depth = fragments.zbuf
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(
            self, meshes_world: Meshes, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.
        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments


class Renderer():
    def __init__(self, img_size, device='cpu'):
        if isinstance(img_size, tuple):
            self.img_size = torch.tensor(img_size).reshape(1, 2).float().cuda()
        else:
            self.img_size = img_size
        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.amblights = AmbientLights(device=device)
        self.point_lights = PointLights(location=[[0, 0, -1.0]], device=device)
        self.rasterizer = MeshRasterizer(raster_settings=self.raster_settings)

        self.renderer_rgb = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=HardPhongShader(device=device)
            # shader=HardGouraudShader(device=device)
            # shader=myShader(device=device)
        )
        self.renderer_depth = MeshRendererWithFragments(
            rasterizer=self.rasterizer,
            shader=myShader(device=device)
        )
        self.device = device

    def build_camera(self, cameras=None,
                     scale=None, trans2d=None):
        if scale is not None and trans2d is not None:
            bs = scale.shape[0]
            R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).repeat(bs, 1, 1).to(scale.dtype)
            T = torch.tensor([0, 0, 10]).repeat(bs, 1).to(scale.dtype)
            return OrthographicCameras(focal_length=2 * scale.to(self.device),
                                       principal_point=-trans2d.to(self.device),
                                       R=R.to(self.device),
                                       T=T.to(self.device),
                                       in_ndc=True,
                                       device=self.device)
        if cameras is not None:
            # cameras: bs x 3 x 3
            fs = -torch.stack((cameras[:, 0, 0], cameras[:, 1, 1]), dim=-1) * 2 / self.img_size
            pps = -cameras[:, :2, -1] * 2 / self.img_size + 1
            return PerspectiveCameras(focal_length=fs.to(self.device),
                                      principal_point=pps.to(self.device),
                                      in_ndc=True,
                                      device=self.device
                                      )

    def build_texture(self, uv_verts=None, uv_faces=None, texture=None,
                      v_color=None):
        if uv_verts is not None and uv_faces is not None and texture is not None:
            return TexturesUV(texture.to(self.device), uv_faces.to(self.device), uv_verts.to(self.device))
        if v_color is not None:
            return TexturesVertex(verts_features=v_color.to(self.device))

    def render(self, verts, faces, cameras, textures, amblights=False, lights=None):
        if lights is None:
            if amblights:
                lights = self.amblights
            else:
                lights = self.point_lights
        mesh = Meshes(verts=verts.to(self.device), faces=faces.to(self.device), textures=textures)
        output = self.renderer_rgb(mesh, cameras=cameras, lights=lights)
        # output = self.renderer_rgb(mesh, cameras=cameras)
        alpha = output[..., 3]
        img = output[..., :3] / 255
        return img, alpha


class mano_renderer(Renderer):
    def __init__(self, mano_path=None, dense_path=None, img_size=256, device='cpu', hand_type='left'):
        super(mano_renderer, self).__init__(img_size, device)
        if mano_path is None:
            mano_path = get_mano_path()
        if dense_path is None:
            dense_path = get_dense_color_path()

        mano_right = ManoLayer(mano_path['right'], center_idx=None)
        mano_left = ManoLayer(mano_path['left'], center_idx=None)

        right_faces = torch.from_numpy(mano_right.get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        left_faces = right_faces[..., [1, 0, 2]]

        if hand_type == 'left':
            self.faces = left_faces
            self.mano = mano_left.to(self.device)
        else:
            self.faces = right_faces
            self.mano = mano_right.to(self.device)

        self.hand_type = hand_type
        with open(dense_path, 'rb') as file:
            dense_coor = pickle.load(file)
        self.dense_coor = torch.from_numpy(dense_coor) * 255

    def render_rgb(self, cameras=None, scale=None, trans2d=None,
                   R=None, pose=None, shape=None, trans=None,
                   v3d=None,
                   uv_verts=None, uv_faces=None, texture=None, v_color=(255, 255, 255),
                   amblights=False):
        if v3d is None:
            v3d, _ = self.mano(R, pose, shape, trans=trans)
        bs = v3d.shape[0]
        vNum = v3d.shape[1]

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, vNum, 3).to(v3d)

        return self.render(v3d, self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(uv_verts, uv_faces, texture, v_color),
                           amblights=amblights)

    def render_densepose(self, cameras=None, scale=None, trans2d=None,
                         R=None, pose=None, shape=None, trans=None,
                         v3d=None):
        if v3d is None:
            v3d, _ = self.mano(R, pose, shape, trans=trans)
        bs = v3d.shape[0]
        vNum = v3d.shape[1]
        v_color = self.dense_coor.unsqueeze(0).expand(bs, vNum, 3).to(v3d)
        return self.render(v3d, self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(v_color=v_color),
                           True)

    def render_mask(self, cameras=None, scale=None, trans2d=None,
                    v3d=None):
        v_color = torch.zeros((778, 3))
        if self.hand_type == 'left':
            v_color[:, 2] = 255
        else:
            v_color[:, 1] = 255
        rgb, mask = self.render_rgb(cameras, scale, trans2d,
                                    v3d=v3d,
                                    v_color=v_color,
                                    amblights=True)
        return rgb


class mano_two_hands_renderer(Renderer):
    def __init__(self, mano_path=None, dense_path=None, img_size=224, device='cpu'):
        super(mano_two_hands_renderer, self).__init__(img_size, device)
        if mano_path is None:
            mano_path = get_mano_path()
        if dense_path is None:
            dense_path = get_dense_color_path()

        self.mano = {'right': ManoLayer(mano_path['right'], center_idx=None),
                     'left': ManoLayer(mano_path['left'], center_idx=None)}
        self.mano['left'].to(self.device)
        self.mano['right'].to(self.device)

        left_faces = torch.from_numpy(self.mano['left'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        right_faces = torch.from_numpy(self.mano['right'].get_faces().astype(np.int64)).to(self.device).unsqueeze(0)
        left_faces = right_faces[..., [1, 0, 2]]

        self.faces = torch.cat((left_faces, right_faces + 778), dim=1)
        self.single_faces = {'left': left_faces,
                             'right': right_faces}

        with open(dense_path, 'rb') as file:
            dense_coor = pickle.load(file)
        self.dense_coor = torch.from_numpy(dense_coor) * 255

    def render_rgb(self, cameras=None, scale=None, trans2d=None,
                   v3d_left=None, v3d_right=None,
                   uv_verts=None, uv_faces=None, texture=None, v_color=None,
                   amblights=False,
                   lights=None):
        bs = v3d_left.shape[0]
        vNum = v3d_left.shape[1]

        if v_color is None:
            v_color = torch.zeros((778 * 2, 3))
            v_color[:778, 0] = 204
            v_color[:778, 1] = 153
            v_color[:778, 2] = 0
            v_color[778:, 0] = 102
            v_color[778:, 1] = 102
            v_color[778:, 2] = 255

        if not isinstance(v_color, torch.Tensor):
            v_color = torch.tensor(v_color)
        v_color = v_color.expand(bs, 2 * vNum, 3).float().to(self.device)

        v3d = torch.cat((v3d_left, v3d_right), dim=1)

        return self.render(v3d,
                           self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(uv_verts, uv_faces, texture, v_color),
                           amblights,
                           lights)

    def render_rgb_orth(self, scale_left=None, trans2d_left=None,
                        scale_right=None, trans2d_right=None,
                        v3d_left=None, v3d_right=None,
                        uv_verts=None, uv_faces=None, texture=None, v_color=None,
                        amblights=False):
        scale = scale_left
        trans2d = trans2d_left

        s = scale_right / scale_left
        d = -(trans2d_left - trans2d_right) / 2 / scale_left.unsqueeze(-1)

        s = s.unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(1)
        v3d_right = s * v3d_right
        v3d_right[..., :2] = v3d_right[..., :2] + d

        # scale = (scale_left + scale_right) / 2
        # trans2d = (trans2d_left + trans2d_right) / 2

        return self.render_rgb(self, scale=scale, trans2d=trans2d,
                               v3d_left=v3d_left, v3d_right=v3d_right,
                               uv_verts=uv_verts, uv_faces=uv_faces, texture=texture, v_color=v_color,
                               amblights=amblights)

    def render_mask(self, cameras=None, scale=None, trans2d=None,
                    v3d_left=None, v3d_right=None):
        v_color = torch.zeros((778 * 2, 3))
        v_color[:778, 2] = 255
        v_color[778:, 1] = 255
        rgb, mask = self.render_rgb(cameras, scale, trans2d,
                                    v3d_left, v3d_right,
                                    v_color=v_color,
                                    amblights=True)
        return rgb

    def render_densepose(self, cameras=None, scale=None, trans2d=None,
                         v3d_left=None, v3d_right=None, ):
        bs = v3d_left.shape[0]
        vNum = v3d_left.shape[1]

        v3d = torch.cat((v3d_left, v3d_right), dim=1)

        v_color = torch.cat((self.dense_coor, self.dense_coor), dim=0)

        return self.render(v3d,
                           self.faces.repeat(bs, 1, 1),
                           self.build_camera(cameras, scale, trans2d),
                           self.build_texture(v_color=v_color.expand(bs, 2 * vNum, 3).to(v3d_left)),
                           True)

    def render_depth(self, cameras=None, scale=None, trans2d=None,
                     v3d_left=None, v3d_right=None):
        bs = v3d_left.shape[0]
        vNum = v3d_left.shape[1]
        v3d = torch.cat((v3d_left, v3d_right), dim=1)
        cameras = self.build_camera(cameras, scale, trans2d)
        faces = self.faces.repeat(bs, 1, 1)
        v_color = torch.zeros((778 * 2, 3))
        textures = self.build_texture(v_color=v_color.expand(bs, 2 * vNum, 3).to(v3d_left))
        mesh = Meshes(verts=v3d.to(self.device), faces=faces.to(self.device), textures=textures)
        _, fragments = self.renderer_depth(mesh, cameras=cameras)
        depth = fragments.zbuf
        return depth

    def render_single_depth(self, cameras=None, scale=None, trans2d=None, v3d=None, hand_type=None):
        bs = v3d.shape[0]
        vNum = v3d.shape[1]
        cameras = self.build_camera(cameras, scale, trans2d)
        faces = self.single_faces[hand_type].repeat(bs, 1, 1)
        v_color = torch.zeros((vNum, 3))
        textures = self.build_texture(v_color=v_color.expand(bs, vNum, 3).to(v3d))
        mesh = Meshes(verts=v3d.to(self.device), faces=faces.to(self.device), textures=textures)
        _, fragments = self.renderer_depth(mesh, cameras=cameras)
        depth = fragments.zbuf
        return depth