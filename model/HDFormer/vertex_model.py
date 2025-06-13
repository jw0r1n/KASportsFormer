import torch
import torch.nn as nn
import numpy as np

from model.HDFormer.HDFormer import HDFormer

# from utils.util import compute_RF_numerical


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *x):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self, logger, writer):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) / 1e6  # Unit is Mega
        logger.info(self)
        logger.info('===>Trainable parameters: %.3f M' % params)
        if writer is not None:
            writer.add_text('Model Summary', 'Trainable parameters: %.3f M' % params)


class Model(BaseModel):

    def __init__(self, skeleton, cfg):
        super(Model, self).__init__()
        self.regress_with_edge = hasattr(cfg, 'regress_with_edge') and cfg.regress_with_edge
        self.backbone = HDFormer(skeleton, cfg)
        num_v, num_e = self.backbone.di_graph.source_M.shape
        self.regressor_type = cfg.regressor_type if hasattr(cfg, 'regressor_type') else 'conv'
        if self.regressor_type == 'conv':
            self.joint_regressor = nn.Conv2d(self.backbone.PLANES[0], 3 * (num_v - 1),
                                             kernel_size=(3, num_v + num_e) if self.regress_with_edge else (3, num_v),
                                             padding=(1, 0), bias=True)
        elif self.regressor_type == 'fc':
            self.joint_regressor = nn.Conv1d(
                self.backbone.PLANES[0] * (num_v + num_e) if self.regress_with_edge else
                self.backbone.PLANES[0] * num_v,
                3 * (num_v - 1),
                kernel_size=3,
                padding=1, bias=True)
        else:
            raise NotImplemented
        self.input_adjust = nn.Linear(in_features=27, out_features=96)

        self.output_adjust = nn.Linear(in_features=96, out_features=27)

    def forward(self, x_v: torch.Tensor, mean_3d: torch.Tensor = None, std_3d: torch.Tensor = None):
        """
        x: shape [B,C,T,V_v]
        """
        # x_v: (1, 2, 96, 17)
        # fv: (1, 16, 48, 16)
        # fe: (1, 16, 96, 17)


        # 1, 27, 17, 2 -> 1, 2, 27, 17 -> 1, 2, 17, 27

        # x_v = x_v.permute(0, 1, 3, 2)
        x_v = x_v.permute(0, 3, 2, 1)
        x_v = self.input_adjust(x_v)
        x_v = x_v.permute(0, 1, 3, 2)


        fv, fe = self.backbone(x_v)
        B, C, T, V = fv.shape
        _, _, _, E = fe.shape

        # import pdb
        # pdb.set_trace()
        # [B,3*(V-1),T,1]
        if self.regressor_type == 'conv':
            pre_joints = self.joint_regressor(
                torch.cat([fv, fe], dim=-1)) if self.regress_with_edge else self.joint_regressor(fv)
        elif self.regressor_type == 'fc':
            x = (torch.cat([fv, fe], dim=-1) if self.regress_with_edge else fv) \
                .permute(0, 1, 3, 2).contiguous().view(B, -1, T)
            pre_joints = self.joint_regressor(x)
        else:
            raise NotImplemented
        pre_joints = pre_joints.view(B, 3, V - 1, T).permute(0, 1, 3, 2).contiguous()  # [B,3,T,V-1]
        pre_joints = torch.cat(
            (torch.zeros((B, 3, T, 1), dtype=pre_joints.dtype, device=pre_joints.device),
             pre_joints),
            dim=-1)
        # pre_joints = pre_joints * std_3d + mean_3d

        # ([1, 3, 96, 17])

        pre_joints = pre_joints.permute(0, 1, 3, 2)
        pre_joints = self.output_adjust(pre_joints)
        # pre_joints = pre_joints.permute(0, 1, 3, 2)
        pre_joints = pre_joints.permute(0, 3, 2, 1)


        return pre_joints
