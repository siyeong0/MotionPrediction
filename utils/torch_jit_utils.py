import torch

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

# returns rotation matrix
# [r00, r10, r20, r01, r11, r21, r02, r12, r22] r<row><column>
@torch.jit.script
def nn_rep_to_matrix(nn):
    # assert nn = [r00, r10, r20, r01, r11, r21]
    c0 = torch.nn.functional.normalize(nn[:, 0:3], dim=1)
    c1 = torch.nn.functional.normalize(nn[:, 3:6], dim=1)
    
    a0 = torch.zeros(nn.shape[0], 3, device=nn.device, dtype=torch.float32)
    a1 = torch.zeros(nn.shape[0], 3, device=nn.device, dtype=torch.float32)
    a2 = torch.zeros(nn.shape[0], 3, device=nn.device, dtype=torch.float32)
    a0[:, 0] = 1.0
    a1[:, 1] = 1.0
    a2[:, 2] = 1.0
    
    r02 = torch.linalg.det(torch.stack((c0,c1,a0), dim=1))
    r12 = torch.linalg.det(torch.stack((c0,c1,a1), dim=1))
    r22 = torch.linalg.det(torch.stack((c0,c1,a2), dim=1))
    c2 = torch.stack((r02, r12, r22), dim=1)
    
    return torch.stack((c0,c1,c2), dim=2)

# returns [roll, pitch, yaw] : euler ZYX
# input tensor shape have to be [..., 3, 3]
@torch.jit.script
def rot_mat_to_vec(m):
    r00 = m[:, 0, 0]
    #r01 = m[:, 0, 1]
    #r02 = m[:, 0, 2]
    r10 = m[:, 1, 0]
    #r11 = m[:, 1, 1]
    #r12 = m[:, 1, 2]
    r20 = m[:, 2, 0]
    r21 = m[:, 2, 1]
    r22 = m[:, 2, 2]
    
    x = torch.atan2(r21, r22)
    y = torch.atan2(-r20, r00*r00+r10*r10)
    z = torch.atan2(r10, r00)
    
    return torch.stack((x,y,z), dim=1)