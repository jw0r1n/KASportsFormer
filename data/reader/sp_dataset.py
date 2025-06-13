import torch
from torch.utils.data import Dataset
import os
import random
import pickle
import copy


class SportsPose3DDataset(Dataset):
    def __init__(self, args_dict, data_split):
        self.model_name = args_dict.model_name
        self.input_channel_number = args_dict.input_channel_number
        self.data_root = args_dict.data_root
        # self.add_velocity = args_dict.add_velocity
        self.flip = args_dict.flip
        # self.use_proj_as_2d = args_dict.use_proj_as_2d
        self.clip_set_name = args_dict.clip_set_name
        self.data_split = data_split
        # self.return_stats = return_stats
        self.clip_list = self._generate_file_list()

    def _generate_file_list(self) -> list:
        clip_list = []
        clip_dir_path = os.path.join(self.data_root, self.clip_set_name, self.data_split)
        clip_file_path_list = sorted(os.listdir(clip_dir_path))
        for i in clip_file_path_list:
            clip_list.append(os.path.join(clip_dir_path, i))
        return clip_list

    def read_pkl_file(self, pkl_file_url):
        file = open(pkl_file_url, 'rb')
        pkl_content = pickle.load(file)
        file.close()
        return pkl_content

    def flip_data(self, data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
        flipped_data = copy.deepcopy(data)
        flipped_data[..., 0] *= -1
        flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
        return flipped_data

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        file_path = self.clip_list[idx]
        joint_file = self.read_pkl_file(file_path)


        if self.data_split == 'train':
            joint_input = joint_file["data_input"]
            joint_label = joint_file["data_label"]
        else:
            # joint_label = joint_file["data_label"]
            joint_input = joint_file["data_input"]
            joint_label_scaled = joint_file["data_label_scaled"]
            joint_factor = joint_file["data_factor"]
            joint_action = joint_file["data_action"]
            joint_res = joint_file["data_res"]
            # joint_env = joint_file["data_env"]



        # if joint_input is None or self.use_proj_as_2d:
        #     joint_input  = self._construct_joint2d_by_projection(joint_label)

        # if self.add_velocity:
        #     joint_2d_coord = joint_input[..., :2]
        #     velocity_joint_2d = joint_2d_coord[1:] - joint_2d_coord[:-1]
        #     joint_input = joint_input[:-1]
        #     joint_input = np.concatenate((joint_input, velocity_joint_2d), axis=-1)
        #
        #     joint_label = joint_label[:-1]

        if self.data_split == 'train':
            if self.flip and random.random() > 0.5:
                joint_input = self.flip_data(joint_input)
                joint_label = self.flip_data(joint_label)

        # if self.return_stats:
        #     return torch.FloatTensor(joint_input), torch.FloatTensor(joint_label), joint_file['mean'], joint_file['std']
        # else:
        #     return torch.FloatTensor(joint_input), torch.FloatTensor(joint_label)

        if self.input_channel_number == 2:
            joint_input = joint_input[:, :, :2]

        if self.data_split == 'train':
            return torch.FloatTensor(joint_input), torch.FloatTensor(joint_label)
        else:
            return (torch.FloatTensor(joint_input), joint_label_scaled, joint_factor,
                    joint_action, joint_res)





