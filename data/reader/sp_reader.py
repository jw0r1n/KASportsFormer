import pickle
import numpy as np
import torch


class DataReaderSportsPose(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, source_file_path, read_confidence=True):
        self.split_id_train = None
        self.split_id_test = None
        self.data_source = self.read_pkl(source_file_path)
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.test_hw = None

    def read_pkl(self, source_file_url):
        file = open(source_file_url, 'rb')
        content = pickle.load(file)
        file.close()
        return content


    def read_2d_sp(self):
        trainset = self.data_source['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.data_source['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.data_source['train']['camera_name']):
            if camera_name == 'outdoors':
                res_w, res_h = 1312, 1216
            elif camera_name == 'indoors':
                res_w, res_h = 1216, 1936
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.data_source['test']['camera_name']):
            if camera_name == 'outdoors':
                res_w, res_h = 1312, 1216
            elif camera_name == 'indoors':
                res_w, res_h = 1216, 1936
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.data_source['train'].keys():
                train_confidence = self.data_source['train']['confidence'][::self.sample_stride].astype(np.float32)
                test_confidence = self.data_source['test']['confidence'][::self.sample_stride].astype(np.float32)
                if len(train_confidence.shape) == 2:  # (n, 17)
                    train_confidence = train_confidence[:, :, None]
                    test_confidence = test_confidence[:, :, None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:, :, 0:1]
                test_confidence = np.ones(testset.shape)[:, :, 0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d_sp(self):
        train_labels = self.data_source['train']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]
        test_labels = self.data_source['test']['joint3d_image'][::self.sample_stride, :, :3].astype(
            np.float32)  # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.data_source['train']['camera_name']):
            if camera_name == 'outdoors':
                res_w, res_h = 1312, 1216
            elif camera_name == 'indoors':
                res_w, res_h = 1216, 1936
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2

        for idx, camera_name in enumerate(self.data_source['test']['camera_name']):
            if camera_name == 'outdoors':
                res_w, res_h = 1312, 1216
            elif camera_name == 'indoors':
                res_w, res_h = 1216, 1936
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2

        return train_labels, test_labels

    def read_hw_sp(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.data_source['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.data_source['test']['camera_name']):
            if camera_name == 'outdoors':
                res_w, res_h = 1312, 1216
            elif camera_name == 'indoors':
                res_w, res_h = 1216, 1936
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw

    def split_clips(self, vid_list, n_frames, data_stride):
        """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
        result = []
        n_clips = 0
        st = 0
        i = 0
        saved = set()
        while i < len(vid_list):
            i += 1
            if i - st == n_frames:
                result.append(range(st, i))
                saved.add(vid_list[i - 1])
                st = st + data_stride
                n_clips += 1
            if i == len(vid_list):
                break
            if vid_list[i] != vid_list[i - 1]:
                if not (vid_list[i - 1] in saved):
                    resampled = self.resample(i - st, n_frames) + st
                    result.append(resampled)
                    saved.add(vid_list[i - 1])
                st = i
        return result

    def mysplit_clips(self, vid_list, n_frames, data_stride):
        result = []
        start = 0
        i = 0
        while i < len(vid_list):
            if vid_list[i] != vid_list[start]:
                if (i - start) >= (n_frames / 2):
                    resampled = self.resample(i - start, n_frames) + start
                    result.append(resampled)
                start = i
                i = i - 1
            else:
                if i - start + 1 == n_frames:
                    result.append(range(start, i + 1))
                    start = start + data_stride
            i = i + 1
        return result



    def resample(self, ori_len, target_len, replay=False, randomness=True):
        """Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
        if replay:
            if ori_len > target_len:
                st = np.random.randint(ori_len - target_len)
                return range(st, st + target_len)  # Random clipping from sequence
            else:
                return np.array(range(target_len)) % ori_len  # Replay padding
        else:
            if randomness:
                even = np.linspace(0, ori_len, num=target_len, endpoint=False)
                if ori_len < target_len:
                    low = np.floor(even)
                    high = np.ceil(even)
                    sel = np.random.randint(2, size=even.shape)
                    result = np.sort(sel * low + (1 - sel) * high)
                else:
                    interval = even[1] - even[0]
                    result = np.random.random(even.shape) * interval + even
                result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
            else:
                result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
            return result

    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.data_source['train']['source'][::self.sample_stride]  # (1559752,)
        vid_list_test = self.data_source['test']['source'][::self.sample_stride]  # (566920,)
        self.split_id_train = self.split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        self.split_id_test = self.split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        # self.split_id_train = self.mysplit_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)
        # self.split_id_test = self.mysplit_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)

        return self.split_id_train, self.split_id_test

    def turn_into_test_clips(self, data):
        """Converts (total_frames, ...) tensor to (n_clips, n_frames, ...) based on split_id_test"""
        split_id_train, split_id_test = self.get_split_id()
        data = data[split_id_test]
        return data

    def get_hw_sp(self):
        #       Only Testset HW is needed for denormalization
        test_hw = self.read_hw_sp()  # train_data (1559752, 2) test_data (566920, 2)
        test_hw = self.turn_into_test_clips(test_hw)[:, 0, :]  # (N, 2)
        return test_hw

    # def get_sliced_data_sp(self):
    #     train_data, test_data = self.read_2d_sp()  # train_data (1559752, 17, 3) test_data (566920, 17, 3)
    #     train_labels, test_labels = self.read_3d_sp()  # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
    #     split_id_train, split_id_test = self.get_split_id()
    #     train_data, test_data = train_data[split_id_train], test_data[split_id_test]  # (N, 27, 17, 3)
    #     train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]  # (N, 27, 17, 3)
    #
    #
    #     return train_data, test_data, train_labels, test_labels

    def get_sliced_data_sp(self):
        train_data, test_data = self.read_2d_sp()  # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d_sp()  # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)

        split_id_train, split_id_test = self.get_split_id()

        train_data, test_data = train_data[split_id_train], test_data[split_id_test]  # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]  # (N, 27, 17, 3)

        # train_action = np.array(self.data_source["train"]["action"])[split_id_train]
        test_action = np.array(self.data_source["test"]["action"])[split_id_test]
        # train_envtag = self.data_source["train"]["camera_name"][split_id_train]
        test_envtag = self.data_source["test"]["camera_name"][split_id_test]
        # train_factor = self.data_source["train"]["2.5d_factor"][split_id_train]
        test_factor = self.data_source["test"]["2.5d_factor"][split_id_test]
        # train_label_scaled = self.data_source["train"]["joints_2.5d_image"][split_id_train]
        test_label_scaled = self.data_source["test"]["joints_2.5d_image"][split_id_test]

        # total_frames_range = np.array(range(len(self.data_source["test"]["joints_2.5d_image"])))
        # frame_index_clips = total_frames_range[split_id_test]


        test_hw = self.get_hw_sp()

        train_dict = {
            "data": train_data,
            "label": train_labels,
            # "action": train_action,
            # "envtag": train_envtag,
            # "factor": train_factor,
            # "label_scaled": train_label_scaled,
        }
        test_dict = {
            "data": test_data,
            "label": test_labels,
            "action": test_action,
            "envtag": test_envtag,
            "factor": test_factor,
            "label_scaled": test_label_scaled,
            "test_hw": test_hw,
            # "frame_index_clips": frame_index_clips,
        }

        # return train_data, test_data, train_labels, test_labels
        return train_dict, test_dict

    def denormalize_sp(self, test_data, all_sequence=False):
        #       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)
        if all_sequence:
            test_data = self.turn_into_test_clips(test_data)

        n_clips = test_data.shape[0]
        test_hw = self.get_hw_sp()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw), f"Data n_clips is {len(data)} while test_hw size is {len(test_hw)}"
        # denormalize (x,y,z) coordinates for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data  # [n_clips, -1, 17, 3]








if __name__ == '__main__':
    sports_data_processor = DataReaderSportsPose(n_frames=27, sample_stride=1, data_stride_train=27 // 3,
                                                 data_stride_test=27,
                                                 source_file_path = '../sp_hr_conf_cam_source_1camera.pkl')
    train_dict, test_dict = sports_data_processor.get_sliced_data_sp()
    print(sports_data_processor)

    temp = torch.randn((16, 27, 17, 3))
    print(len(temp))

