import os
from tqdm import tqdm
import argparse
import pickle
import sys

sys.path.append('..')
from reader.sp_reader import DataReaderSportsPose

# def save_clips(set_name, root_path, input_set, label_set): # deprecated
#     assert len(input_set) == len(label_set), "wait, what. length of input is not equal to label??"
#     len_total = len(input_set)
#     save_path = os.path.join(root_path, set_name)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     loop = tqdm(range(len_total))
#     loop.set_description(f"clips {set_name} generate...")
#     for i in loop:
#         data_input, data_label = input_set[i], label_set[i]
#         data_dict = {
#             "data_input": data_input,
#             "data_label": data_label
#         }
#         with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as file:
#             pickle.dump(data_dict, file)


def save_clips_train(root_path, input_set, label_set, root_rel: bool = True):
    assert len(input_set) == len(label_set), "wait, what. length of input is not equal to label??"
    clip_num = len(input_set)
    save_path = os.path.join(root_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loop = tqdm(range(clip_num))
    loop.set_description(f"train clips generating...")
    for i in loop:
        data_input = input_set[i]
        data_label = label_set[i]
        if root_rel:  # (N, 27, 17, 3)
            data_label = data_label - data_label[..., 0:1, :]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label,
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as file:
            pickle.dump(data_dict, file)

def save_clips_test(root_path, input_set, label_set, label_scaled_set, action_set, envtag_set, factor_set, hw_set):
    assert len(input_set) == len(label_scaled_set), "wait, what. length of input is not equal to label??"
    clip_num = len(input_set)
    save_path = os.path.join(root_path, "test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    loop = tqdm(range(clip_num))
    loop.set_description(f"test clips generating...")
    for i in loop:
        data_input = input_set[i]
        data_label = label_set[i]
        data_label_scaled = label_scaled_set[i]
        data_factor = factor_set[i]
        action = set(action_set[i])
        assert len(action) == 1, f"wait, clip index {i} contains more than one action ??"
        action = list(action)[0]
        envtag = set(envtag_set[i])
        assert len(envtag) == 1, f"wait, clip index {i} contains more than one environment tag ??"
        envtag = list(envtag)[0]
        res_wh = hw_set[i]  # (res_w, res_h)

        data_dict = {
            "data_input": data_input,
            "data_label": data_label,
            "data_label_scaled": data_label_scaled,
            "data_factor": data_factor,
            "data_res": res_wh,
            "data_action": action,
            "data_env": envtag,
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as file:
            pickle.dump(data_dict, file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=27)
    parser.add_argument("--data-type", type=str, default="det")
    args = parser.parse_args()
    n_frames = args.n_frames
    if args.data_type == "det":
        data_type = "dete"
        source_file = '../sp_hr_conf_cam_source_1camera.pkl'
    elif args.data_type == "gt":
        data_type = "gt"
        source_file = '../sp_no_conf_cam_source_final.pkl'
    else:
        raise ValueError(f"Unknown data type: {args.data_type}. Use 'det' or 'gt'.")


    sports_data_processor = DataReaderSportsPose(n_frames=n_frames, sample_stride=1, data_stride_train=n_frames // 3,
                                                 data_stride_test=n_frames,
                                                source_file_path = source_file)
    train_dict, test_dict = sports_data_processor.get_sliced_data_sp()

    print(f"Now generating clips ({n_frames} frames):...")
    print(f"train input shape: {train_dict['data'].shape}")
    print(f"train label shape: {train_dict['label'].shape}")
    print(f"test input shape: {test_dict['data'].shape}")
    print(f"test label shape: {test_dict['label'].shape}")

    root_path = f"../clips/SP{data_type}-{n_frames}"
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # save_clips(set_name="train", root_path=root_path, input_set=train_input, label_set=train_label)
    # save_clips(set_name="test", root_path=root_path, input_set=test_input, label_set=test_label)
    save_clips_train(root_path=root_path, input_set=train_dict["data"], label_set=train_dict["label"])
    save_clips_test(root_path=root_path, input_set=test_dict["data"], label_set=test_dict["label"], label_scaled_set=test_dict["label_scaled"],
                    action_set=test_dict["action"], envtag_set=test_dict["envtag"],
                    factor_set=test_dict["factor"], hw_set=test_dict["test_hw"])



if __name__ == '__main__':
    main()


