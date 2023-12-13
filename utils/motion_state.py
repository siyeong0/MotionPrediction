import torch

class MotionState:
    def __init__(self, state, num_dofs, num_bodies):
        self.state = state
        self.num_dofs = num_dofs
        self.num_bodies = num_bodies
    
    @property
    def root_state(self):
        return self.state[:, 0:13]
    @property
    def root_pos(self):
        return self.root_state[:, 0:3]
    @property
    def root_rot(self):
        return self.root_state[:, 3:7]
    @property
    def root_vel(self):
        return self.root_state[:, 7:10]
    @property
    def root_ang_vel(self):
        return self.root_state[:, 10:13]
    
    @property
    def dof_state(self):
        return self.state[:, 13:13+2*self.num_dofs].view(self.state.shape[0], self.num_dofs, 2)
    @property
    def dof_pos(self):
        return self.dof_state[:,:,0]
    @property
    def dof_vel(self):
        return self.dof_state[:,:,1]
    
    @property
    def link_state(self):
        return torch.cat((self.link_pos, self.link_rot, self.link_vel), dim=-1)
    @property
    def link_pos(self):
        return self.state[:, 13+2*self.num_dofs:13+2*self.num_dofs+3*self.num_bodies].view(self.state.shape[0], self.num_bodies, 3)
    @property
    def link_rot(self):
        return self.state[:, 13+2*self.num_dofs+3*self.num_bodies:13+2*self.num_dofs+7*self.num_bodies].view(self.state.shape[0], self.num_bodies, 4)
    @property
    def link_vel(self):
        return self.state[:, 13+2*self.num_dofs+7*self.num_bodies:13+2*self.num_dofs+10*self.num_bodies].view(self.state.shape[0], self.num_bodies, 3)