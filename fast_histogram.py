import torch
import math
import numpy as np


def histogram(img: torch.Tensor, mask: torch.Tensor, channels = [32, 32, 32], normalize=True) -> torch.Tensor:
    """
    Returns a color histogram of an input image

    Args:
        img: (H, W, 3) or (B, H, W, 3) torch tensor containing RGB values
        mask: (H, W) or (B, H, W) torch tensor with mask values
        channels: List of length 3 containing number of bins per each channel
        normalize: If True, normalizes histogram

    Returns:
        hist: Histogram of shape (*channels)
    """

    # Make the color of an image to be in range (0, 255)
    tgt_img = img.clone().detach()
    final_mask = mask.clone().detach()
    max_rgb = torch.LongTensor([255] * 3).to(tgt_img.device)
    bin_size = torch.ceil(max_rgb.float() / torch.tensor(channels).float().to(tgt_img.device)).long()

    if tgt_img.max() <= 1:
        tgt_img = (tgt_img * max_rgb.reshape(-1, 3)).long()
    
    if len(img.shape) == 3:
        tgt_rgb = tgt_img[torch.nonzero(final_mask.long(), as_tuple=True)].long() # (N, 3) torch tensor
        tgt_rgb = tgt_rgb // bin_size.reshape(-1, 3)

        tgt_rgb = tgt_rgb[:, 0] + channels[0] * tgt_rgb[:, 1] + channels[0] * channels[1] * tgt_rgb[:, 2]

        hist = torch.bincount(tgt_rgb, minlength=channels[0] * channels[1] * channels[2]).float()
        hist = hist.reshape(*channels)

        if normalize:
            # normalize histogram
            hist = hist / hist.sum()
    else:  # Batched input
        eps = 1e-6
        tgt_img = tgt_img // bin_size.reshape(-1, 3)
        tgt_img = tgt_img[..., 0] + channels[0] * tgt_img[..., 1] + channels[0] * channels[1] * tgt_img[..., 2]  # (B, H, W)
        tgt_img *= final_mask.float()
        tgt_img = tgt_img.reshape(tgt_img.shape[0], -1).long()  # (B, H * W)
        hist = torch.zeros([tgt_img.shape[0], channels[0] * channels[1] * channels[2]], device=tgt_img.device, dtype=torch.long).scatter_add(
            dim=-1, index=tgt_img, src=torch.ones_like(tgt_img, dtype=torch.long))  # (B, C)
        hist[:, 0] -= (~final_mask).reshape(tgt_img.shape[0], -1).sum(-1)  # Subtract zeros from final mask
        hist = hist.float()

        if normalize:
            hist_sum = hist.sum(-1)
            hist = hist / (hist_sum.reshape(-1, 1) + eps)  # Normalize
        hist = hist.reshape([hist.shape[0], *channels])

    return hist


def warp_from_img(img: torch.Tensor, coord_arr: torch.Tensor, padding='zeros', mode='bilinear') -> torch.Tensor:
    """
    Image warping function
    Use coord_arr as a grid for warping from img

    Args:
        img: (H, W, C) torch tensor containing image RGB values
        coord_arr: (H, W, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid

    Returns:
        sample_rgb: (H, W, C) torch tensor containing sampled RGB values
    """

    img = img.permute(2, 0, 1)  # (C, H, W)
    img = torch.unsqueeze(img, 0)  # (1, C, H, W)

    # sampling from img
    sample_arr = coord_arr.unsqueeze(0)  # (1, H, W, 2)
    sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
    sample_rgb = F.grid_sample(img, sample_arr, align_corners=False, padding_mode=padding, mode=mode)  # (1, C, H, W)

    sample_rgb = sample_rgb.squeeze(0).permute(1, 2, 0)  # (H, W, C)

    return sample_rgb


def cloud2idx(xyz: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """
    Change 3d coordinates to image coordinates ranged in [-1, 1].

    Args:
        xyz: (N, 3) torch tensor containing xyz values of the point cloud data
        batched: If True, performs batched operation with xyz considered as shape (B, N, 3)

    Returns:
        coord_arr: (N, 2) torch tensor containing transformed image coordinates
    """
    if batched:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[..., :2], dim=-1)), xyz[..., 2] + 1e-6), -1)  # (B, N, 1)

        # horizontal angle
        phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1] + 1e-6)  # (B, N, 1)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)  # (B, N, 2)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[..., 0] / (np.pi * 2), sphere_cloud_arr[..., 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)  # (B, N, 2)

    else:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

        # horizontal angle
        phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)

    return coord_arr


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw.unsqueeze(0)
        pitch = pitch.unsqueeze(0)
        roll = roll.unsqueeze(0)

        tensor_0 = torch.zeros(1, device=yaw.device)
        tensor_1 = torch.ones(1, device=yaw.device)

        RX = torch.stack([
                        torch.stack([tensor_1, tensor_0, tensor_0]),
                        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        return R
    
    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return torch.stack(tot_mtx)


# Code excerpted from https://github.com/haruishi43/equilib
def create_coordinate(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    r"""Create mesh coordinate grid with height and width

    return:
        coordinate: numpy.ndarray
    """
    xs = torch.linspace(0, w_out - 1, w_out, device=device)
    theta = np.pi - xs * 2 * math.pi / w_out
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    phi, theta = torch.meshgrid([phi, theta])
    coord = torch.stack((theta, phi), axis=-1)
    return coord


def create_area_mask(h_out: int, w_out: int, device=torch.device('cpu')) -> np.ndarray:
    """
    Create H x W numpy array containing area size for each location in the gridded sphere: dtheta x dphi x sin(theta)
    """
    ys = torch.linspace(0, h_out - 1, h_out, device=device)
    phi = ys * math.pi / h_out + np.pi / (h_out * 2)
    return torch.stack([phi] * w_out, dim=1)  # h_out * w_out


def compute_sampling_grid(ypr, num_split_h, num_split_w, inverse=False):
    """
    Utility function for computing sampling grid using yaw, pitch, roll
    We assume the equirectangular image to be splitted as follows:

    -------------------------------------
    |   0    |   1    |    2   |    3   |
    |        |        |        |        |
    -------------------------------------
    |   4    |   5    |    6   |    7   |
    |        |        |        |        |
    -------------------------------------

    Indices are assumed to be ordered in compliance to the above convention.
    Args:
        ypr: torch.tensor of shape (3, ) containing yaw, pitch, roll
        num_split_h: Number of horizontal splits
        num_split_w: Number of vertical splits
        inverse: If True, calculates sampling grid with inverted rotation provided from ypr

    Returns:
        grid: Sampling grid for generating rotated images according to yaw, pitch, roll
    """
    if inverse:
        R = rot_from_ypr(ypr)
    else:
        R = rot_from_ypr(ypr).T

    H, W = num_split_h, num_split_w
    a = create_coordinate(H, W, ypr.device)
    a[..., 0] -= np.pi / (num_split_w)  # Add offset to align sampling grid to each pixel center
    a[..., 1] += np.pi / (num_split_h * 2)  # Add offset to align sampling grid to each pixel center
    norm_A = 1
    x = norm_A * torch.sin(a[:, :, 1]) * torch.cos(a[:, :, 0])
    y = norm_A * torch.sin(a[:, :, 1]) * torch.sin(a[:, :, 0])
    z = norm_A * torch.cos(a[:, :, 1])
    A = torch.stack((x, y, z), dim=-1)  # (H, W, 3)
    _B = R @ A.unsqueeze(3)
    _B = _B.squeeze(3)
    grid = cloud2idx(_B.reshape(-1, 3)).reshape(H, W, 2)
    return grid


def fast_histogram_generation(img: torch.Tensor, rot: torch.Tensor, num_split_h: int, num_split_w: int, num_bins: list = [8, 8, 8]):
    """
    Fast histogram generation for arbitrary rotations given a single panorama image in equirectangular projection

    Args:
        img: (H, W, 3) torch tensor containing RGB values of the image
        rot: (K, 3) torch tensor containing K rotations with (yaw, pitch, roll) components
        num_split_h: Number of split along horizontal direction
        num_split_w: Number of split along vertical direction
        num_bins: Number of bins for generating color histograms

    Returns:
        hist: (K, num_split_h, num_split_w, *num_bins) torch tensor containing color histograms at each rotation
    """
    img = img.clone().detach() * 255
    # masking coordinates to remove pixels whose RGB value is [0, 0, 0]
    img_mask = torch.zeros([img.shape[0], img.shape[1]], dtype=torch.bool, device=img.device)
    img_mask[torch.sum(img == 0, dim=2) != 3] = True

    img_chunk = []
    for img_hor_chunk in torch.chunk(img, num_split_h, dim=0):
        img_chunk += [*torch.chunk(img_hor_chunk, num_split_w, dim=1)]

    img_chunk = torch.stack(img_chunk, dim=0)  # (B, H, W, C)
    img_mask_chunk = torch.zeros(img_chunk.shape[0], img_chunk.shape[1], img_chunk.shape[2], dtype=torch.bool, device=xyz.device)
    img_mask_chunk[torch.sum(img_chunk == 0, dim=-1) != 3] = True

    # Initialize grid sample locations
    grid_list = [compute_sampling_grid(ypr, num_split_h, num_split_w) for ypr in rot]

    orig_img_hist = histogram(img_chunk, img_mask_chunk, num_bins)  # (num_split_h * num_split_w, num_bins[0], ...)
    orig_img_hist = orig_img_hist.reshape(num_split_h, num_split_w, -1)

    hist = []
    for j in range(len(rot)):
        rot_hist = warp_from_img(orig_img_hist, grid_list[j], padding='reflection', mode='nearest').reshape(num_split_h, num_split_w, *num_bins)
        hist.append(rot_hist)
    
    hist = torch.stack(hist, dim=0)

    return hist
