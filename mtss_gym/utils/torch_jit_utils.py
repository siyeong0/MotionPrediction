import torch

from isaacgymenvs.utils.torch_jit_utils import quat_from_angle_axis

@torch.jit.script
def exp_neg_norm_square(coef: float, tensor: torch.Tensor) -> torch.Tensor:
    return torch.exp(-coef * torch.pow(torch.norm(tensor, dim=-1), 2))

# returns first and second columns of 3x3 rotation matrix
# On the Continuity of Rotation Representations in Neural Networks: https://arxiv.org/abs/1812.07035
@torch.jit.script
def quat_to_nn_rep(q):
    # Quaternion to Rotation Matrix: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    q0 = q[:,0].unsqueeze(1)
    q1 = q[:,1].unsqueeze(1)
    q2 = q[:,2].unsqueeze(1)
    q3 = q[:,3].unsqueeze(1)
    
    r00 = 1. - 2. * (q1 * q1 + q2 * q2)
    r01 = 2. * (q0 * q1 - q2 * q3)
    #r02 = 2. * (q0 * q2 + q1 * q3)
    
    r10 = 2 * (q0 * q1 + q2 * q3)
    r11 = 1.0 - 2 * (q0 * q0 + q2 * q2)
    #r12 = 2 * (q1 * q2 - q0 * q3)
    
    r20 = 2 * (q0 * q2 - q1 * q3)
    r21 = 2 * (q1 * q2 + q0 * q3)
    #r22 = 1.0 - 2 * (q0 * q0 + q1 * q1)
    
    return torch.cat((r00, r10, r20, r01, r11, r21), dim=1)