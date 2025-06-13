import torch
from torch import nn



def mpjpe_loss_calc(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert predict.shape == target.shape, 'shape unaligned (mpjpe_loss_calc)'    # (N, T, 17, 3)
    interval1 = torch.norm(predict - target, dim=len(target.shape) - 1)  # (N, T, 17)
    interval2 = torch.mean(interval1)   # () one single value
    return interval2


def n_mpjpe_loss_calc(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert predict.shape == target.shape, 'shape unaligned (n_mpjpe_loss_calc)'    # (N, T, 17, 3)
    norm_predicted = torch.mean(torch.sum(predict ** 2, dim=3, keepdim=True), dim=2, keepdim=True) # 上下两边都扩大到了 predict倍
    norm_target = torch.mean(torch.sum(target * predict, dim=3, keepdim=True), dim=2, keepdim=True)  # 只是在每一帧里面  (16, 27, 17, 1) -> (16, 27, 1, 1)
    scale = norm_target / norm_predicted  # 将predict scale到 target差不多的
    return mpjpe_loss_calc(scale * predict, target)


def velocity_loss_calc(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert predict.shape == target.shape, 'shape unaligned (velocity_loss_calc)'
    if predict.shape[1] <= 1:  # 没有Temporal长度的时候
        return torch.FloatTensor(1).fill_(0.)[0].to(predict.device)
    velocity_predict = predict[:, 1:] - predict[:, :-1]  # (16, 26, 17, 3)
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predict - velocity_target, dim=-1))  # (16, 26, 17) -> () one single value


def get_limb_lens(x: torch.Tensor) -> torch.Tensor:  # 获取肢体的长度，16个肢体部分
    # input: (N, T, 17, 3)
    # output: (N, T, 16)
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]    # 16 组数值对
    limbs = x[:, :, limbs_id, :]  # (16, 27, 16, 2, 3)
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]  # (16, 27, 16, 3)
    limb_len = torch.norm(limbs, dim=-1)
    return limb_len


def loss_limb_var_calc(x: torch.Tensor) -> torch.Tensor:  # 肢体长度变化的方差的loss, 这个不需要y只需要x
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_len = get_limb_lens(x)
    limb_len_var = torch.var(limb_len, dim=1)
    limb_loss_var = torch.mean(limb_len_var)
    return limb_loss_var  # 但是为什么要方差


def loss_limb_len_calc(predict: torch.Tensor, target: torch.Tensor):
    # input: (N, T, 17, 3), (N, T, 17, 3)
    limb_len_x = get_limb_lens(predict) # (N, T, 16)
    limb_len_gt = get_limb_lens(target)
    return nn.L1Loss()(limb_len_x, limb_len_gt)  # TODO: 感觉这里不对，每次都创建了一个新的Loss实例


def get_limb_cos_simi(predict: torch.Tensor):
    # input (N, T, 17, 3) output (N, T, 16)
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]
    angle_id = [[0, 3], [0, 6], [3, 6], [0, 1], [1, 2],
                [3, 4], [4, 5], [6, 7], [7, 10], [7, 13],
                [8, 13], [10, 13], [7, 8], [8, 9], [10, 11],
                [11, 12], [13, 14], [14, 15]] # 18组
    eps = 1e-7
    limbs = predict[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    limb_angle = limbs[:, :, angle_id, :]  # (16, 27, 18, 2, 3)
    angle_cos = nn.functional.cosine_similarity(limb_angle[:, :, :, 0, :], limb_angle[:, :, :, 1, :], dim=-1) # (16, 27, 18)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))  # (16, 27, 18)

def loss_cos_simi_calc(predict: torch.Tensor, target: torch.Tensor):
    limb_cos_simi_predict = get_limb_cos_simi(predict)
    limb_cos_simi_target = get_limb_cos_simi(target)
    return nn.L1Loss()(limb_cos_simi_predict, limb_cos_simi_target) # TODO: 感觉这里不对，每次都创建了一个新的Loss实例


def loss_cos_simi_velocity_calc(predict: torch.Tensor, target: torch.Tensor):
    assert predict.shape == target.shape, 'shape is not aligned, loss_cos_simi_velocity'
    if predict.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predict.device)
    predict_cos_simi = get_limb_cos_simi(predict)
    target_cos_simi = get_limb_cos_simi(target)
    predict_cos_simi_velocity = predict_cos_simi[:, 1:] - predict_cos_simi[:, :-1]
    target_cos_simi_velocity = target_cos_simi[:, 1:] - target_cos_simi[:, :-1]
    return nn.L1Loss()(predict_cos_simi_velocity, target_cos_simi_velocity)  # TODO: 感觉这里不对，每次都创建了一个新的Loss实例

def loss_2d_weighted(predicted, target, conf):  # 从motionbert里面的，这个是比较好用的weighted
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))

def weighted_mpjpe(predicted, target, w=None):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    if w is None:
        w = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
        # w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))  # (16, 27, 17)


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape) - 1))


if __name__ == '__main__':
    tempx = torch.rand(16, 27, 17, 3)
    tempy = torch.rand(16, 27, 17, 3)

    print(torch.norm(tempy - tempx, dim=len(tempy.shape)-1).shape)


    exit()
    # interval = torch.sum(tempx ** 2, dim=3, keepdim=True)
    # interval2 = torch.mean(interval, dim=2, keepdim=True)
    # tempx_velocity = tempx[:, 1:] - tempx[:, :-1]
    # interval = torch.norm(tempx_velocity, dim=-1)
    # print(tempx_velocity.shape)
    # print(interval.shape)
    # interval2 = torch.mean(interval)
    # print(interval2)
    # print(interval2.shape)
    limbs_id = [[0, 1], [1, 2], [2, 3],
                [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13],
                [8, 14], [14, 15], [15, 16]
                ]
    print(len(limbs_id))
    interval = tempx[:, :, limbs_id, :]
    print(interval.shape)
    interval = interval[:, :, :, 0, :] - interval[:, :, :, 1, :]

    angle_id = [[0, 3], [0, 6], [3, 6], [0, 1], [1, 2],
                [3, 4], [4, 5], [6, 7], [7, 10], [7, 13],
                [8, 13], [10, 13], [7, 8], [8, 9], [10, 11],
                [11, 12], [13, 14], [14, 15]]
    angles = interval[:, :, angle_id, :]
    print(angles.shape)
    angle_cos = nn.functional.cosine_similarity(angles[:, :, :, 0, :], angles[:, :, :, 1, :], dim=-1)
    eps = 1e-7
    interval2 = torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))
    print(interval2.shape)




class Norm_Loss(nn.Module):
    def __init__(self, vec_length_list, eps=1e-6, normalize=False, diff_order='L1'):
        super().__init__()
        self.vec_length_list = vec_length_list
        self.eps = eps
        self.normalize = normalize
        self.diff_order = diff_order

    def normal(self, pose, vec_length):
        # get surface
        start = pose[:, :, :-vec_length, :]
        end = pose[:, :, vec_length:, :]

        x = start[:, 1, :, :] * end[:, 2, :, :] - \
            start[:, 2, :, :] * end[:, 1, :, :]

        y = \
            start[:, 2, :, :] * end[:, 0, :, :] - \
            start[:, 0, :, :] * end[:, 2, :, :]

        z = \
            start[:, 0, :, :] * end[:, 1, :, :] - \
            start[:, 1, :, :] * end[:, 0, :, :]

        # pred_3d_norm = torch.cat([pred_3d_x, pred_3d_y, pred_3d_z], dim=2)
        # pred_3d_norm = torch.cat([pred_3d_x, pred_3d_y, pred_3d_z], dim=1)
        # import pdb
        # pdb.set_trace()
        norm = torch.stack([x, y, z], dim=1)
        if self.normalize:
            norm /= (self.eps + torch.norm(norm, dim=1, keepdim=True))
        return norm

    def forward(self, pred_3d, gt_3d):
        """
        shape is BCTV
        """
        errors = []
        for vec_length in self.vec_length_list:
            norm_pred = self.normal(pred_3d, vec_length)
            norm_gt = self.normal(gt_3d, vec_length)
            err = norm_pred - norm_gt
            if self.diff_order == 'L1':
                err = torch.abs(err).mean()
            elif self.diff_order == 'L2':
                err = torch.norm(err, dim=1).mean()
            errors.append(err)
        return sum(errors) / len(errors)


