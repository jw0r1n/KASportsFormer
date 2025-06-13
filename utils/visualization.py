import torch
import numpy as np
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqdm import tqdm
from utils.utilities import yaml_config_reader
from model.model_tools import load_model


CONNECTIONS = [
(10, 9),
(9, 8),
(8, 11),
(8, 14),
(14, 15),
(15, 16),
(11, 12),
(12, 13),
(8, 7),
(7, 0),
(0, 4),
(0, 1),
(1, 2),
(2, 3),
(4, 5),
(5, 6)
]

def clip_action_print() -> None:
    clip_dir = r'./data/clips/SPgt-27/test/'
    clip_list = []
    clip_file_path_list = sorted(os.listdir(clip_dir))
    for i in clip_file_path_list:
        clip_list.append(os.path.join(clip_dir, i))

    log_file = open('./spgt_clip_action.txt', 'w')

    for single_file_index in tqdm(range(len(clip_list))):
        with open(clip_list[single_file_index], 'rb') as f:
            single_data = pickle.load(f)
        print(clip_file_path_list[single_file_index] + "  " + single_data["data_action"], file=log_file)

    log_file.close()


def visual_clip_generate(args_dict, clip_root_dir_path, clip_save_dir_path, model_path) -> None:
    print(f"model {args_dict.model_name} predicted clip generating... ")
    clip_list = []
    clip_file_path_list = sorted(os.listdir(clip_root_dir_path))
    for i in clip_file_path_list:
        clip_list.append(os.path.join(clip_root_dir_path, i))

    if not os.path.exists(clip_save_dir_path):
        os.makedirs(clip_save_dir_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"current device: {device}")

    model = load_model(args_dict)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"model {args_dict.model_name} checkpoint loaded")
    else:
        raise Exception('check your dumbass checkpoint path')

    model.eval()

    for single_file_index in tqdm(range(len(clip_list))):
        with open(clip_list[single_file_index], "rb") as f:
            single_data = pickle.load(f)

        input_x = single_data["data_input"]
        # input_x = input_x[:, :, :2] # FOR D3DP
        input_x = torch.FloatTensor(input_x)
        input_x = input_x.unsqueeze(dim=0)
        input_x = input_x.to(device)

        # input_label = single_data["data_label"]
        # input_label = torch.FloatTensor(input_label)
        # input_label = input_label.unsqueeze(dim=0)
        # input_label = input_label.to(device)

        with torch.no_grad():
            predicted_result = model(input_x)
            # predicted_result = model(input_x, input_label)

            predicted_result[:, :, 0, :] = 0

        predicted_result = predicted_result.squeeze()
        predicted_result = predicted_result.cpu().numpy()

        res_w, res_h = single_data["data_res"]
        predicted_result[:, :, :2] = (predicted_result[:, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        predicted_result[:, :, 2:] = predicted_result[:, :, 2:] * res_w / 2

        factor = single_data["data_factor"]
        factor = factor[:, None, None]
        ground_truth = single_data["data_label_scaled"]
        predicted_result *= factor

        predicted_result = predicted_result - predicted_result[:, 0:1, :]
        ground_truth = ground_truth - ground_truth[:, 0:1, :]

        single_data["predicted_result"] = predicted_result
        single_data["ground_truth"] = ground_truth

        with open(os.path.join(clip_save_dir_path, clip_file_path_list[single_file_index]), "wb") as file:
            pickle.dump(single_data, file)

def plot_one_figure(max_value, min_value, plot_predict, plot_gt, save_path) -> None:

    x_predict = plot_predict[:, 0]
    y_predict = plot_predict[:, 1]
    z_predict = plot_predict[:, 2]

    x_gt = plot_gt[:, 0]
    y_gt = plot_gt[:, 1]
    z_gt = plot_gt[:, 2]

    fig = plt.figure(figsize=(5,5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.set_xlim3d([min_value[0], max_value[0]])
    ax.set_ylim3d([min_value[1], max_value[1]])
    ax.set_zlim3d([min_value[2], max_value[2]])

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

    for connection in CONNECTIONS:
        start = plot_gt[connection[0], :]
        end = plot_gt[connection[1], :]
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, c='gray')

    for connection in CONNECTIONS:
        start = plot_predict[connection[0], :]
        end = plot_predict[connection[1], :]
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, c='skyblue')

    ax.scatter(x_predict, y_predict, z_predict, c='skyblue')
    ax.scatter(x_gt, y_gt, z_gt, c='gray')

    ax.view_init(elev=20, azim=60)
    plt.savefig(save_path)
    plt.close()



def plot_one_clip(clip_path: str, save_dir: str, clip_file_name: str, reference_path_list: list) -> None:
    with open(clip_path, "rb") as f:
        clip_data = pickle.load(f)

    plot_predict = clip_data["predicted_result"]
    plot_gt = clip_data["ground_truth"]

    plot_predict_all = plot_predict
    plot_gt_all = plot_gt
    for path in reference_path_list:
        temp_clip_path = os.path.join(path, clip_file_name)
        with open(temp_clip_path, "rb") as f:
            temp_clip = pickle.load(f)
        plot_predict_all = np.concatenate((plot_predict_all, temp_clip["predicted_result"]), axis=0)
        plot_gt_all = np.concatenate((plot_gt_all, temp_clip["ground_truth"]), axis=0)


    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    plot_predict = plot_predict @ cam2real
    plot_gt = plot_gt @ cam2real

    plot_predict_all = plot_predict_all @ cam2real
    plot_gt_all = plot_gt_all @ cam2real


    # min_value_predict = np.min(plot_predict, axis=(0, 1))
    # max_value_predict = np.max(plot_predict, axis=(0, 1))
    # min_value_gt = np.min(plot_gt, axis=(0, 1))
    # max_value_gt = np.max(plot_gt, axis=(0, 1))
    #
    # max_value = np.maximum(max_value_predict, max_value_gt)
    # min_value = np.minimum(min_value_predict, min_value_gt)

    min_value_predict = np.min(plot_predict_all, axis=(0, 1))
    max_value_predict = np.max(plot_predict_all, axis=(0, 1))
    min_value_gt = np.min(plot_gt_all, axis=(0, 1))
    max_value_gt = np.max(plot_gt_all, axis=(0, 1))

    max_value = np.maximum(max_value_predict, max_value_gt)
    min_value = np.minimum(min_value_predict, min_value_gt)

    for frame_index in range(plot_predict.shape[0]):
        figure_name = os.path.join(save_dir, f"frame_{frame_index}.png")
        plot_one_figure(max_value, min_value, plot_predict[frame_index], plot_gt[frame_index], figure_name)


def visualization_plot(source_clip_dir, plot_save_dir, reference_clip_path_list) -> None:
    clip_list = []
    clip_file_path_list = sorted(os.listdir(source_clip_dir))
    for i in clip_file_path_list:
        clip_list.append(os.path.join(source_clip_dir, i))

    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    for clip_index in tqdm(range(len(clip_list))):
        with open(clip_list[clip_index], "rb") as f:
            clip_action = pickle.load(f)["data_action"]

        clip_name = clip_file_path_list[clip_index].split('.')[0] + f"_{clip_action}"
        plot_dir = os.path.join(plot_save_dir, clip_name)

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_one_clip(clip_list[clip_index], plot_dir, clip_file_path_list[clip_index], reference_clip_path_list)



def compare_log_print(sports_clip_dir, magf_clip_dir, d3dp_clip_dir) -> None:
    log_file = open('./clip_compare.txt', 'w')
    sports_clip_list = []
    magf_clip_list = []
    d3dp_clip_list = []
    clip_file_path_list = sorted(os.listdir(sports_clip_dir))

    for i in clip_file_path_list:
        sports_clip_list.append(os.path.join(sports_clip_dir, i))
        magf_clip_list.append(os.path.join(magf_clip_dir, i))
        d3dp_clip_list.append(os.path.join(d3dp_clip_dir, i))


    for clip_index in tqdm(range(len(sports_clip_list))):
        with open(sports_clip_list[clip_index], "rb") as f1:
            sports_clip_data = pickle.load(f1)
            # clip_action = pickle.load(f)["data_action"]
        with open(magf_clip_list[clip_index], "rb") as f2:
            magf_clip_data = pickle.load(f2)
        with open(d3dp_clip_list[clip_index], "rb") as f3:
            d3dp_clip_data = pickle.load(f3)

        sports_predict = sports_clip_data["predicted_result"]
        sports_gt = sports_clip_data["ground_truth"]
        magf_predict = magf_clip_data["predicted_result"]
        magf_gt = magf_clip_data["ground_truth"]
        d3dp_predict = d3dp_clip_data["predicted_result"]
        d3dp_gt = d3dp_clip_data["ground_truth"]

        sports_mpjpe = np.mean(np.linalg.norm(sports_predict - sports_gt, axis=2), axis=1)
        magf_mpjpe = np.mean(np.linalg.norm(magf_predict - magf_gt, axis=2), axis=1)
        d3dp_mpjpe = np.mean(np.linalg.norm(d3dp_predict - d3dp_gt, axis=2), axis=1)

        print(clip_file_path_list[clip_index], file=log_file)
        print(sports_clip_data["data_action"], file=log_file)

        for frame_index in range(sports_predict.shape[0]):
            if sports_mpjpe[frame_index] < magf_mpjpe[frame_index] and sports_mpjpe[frame_index] < d3dp_mpjpe[frame_index]:
                print(f"frame {frame_index} sports: {sports_mpjpe[frame_index]:.2f} magf: {magf_mpjpe[frame_index]:.2f} d3dp: {d3dp_mpjpe[frame_index]:.2f}  sports is best", file=log_file)

        print("", file=log_file)

    log_file.close()



def main() -> None:
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--config-path", type=str, default="./configs/sportspose-magf-base.yaml", help="path to the config file")
    # # parser.add_argument("--config-path", type=str, default="configs/sportspose-d3dp.yaml", help="path to the config file")
    # parser.add_argument("--config-path", type=str, default="configs/sportspose-kasportsformer.yaml", help="path to the config file")
    # cli_args = parser.parse_args()
    # args_dict = yaml_config_reader(cli_args.config_path)
    #
    # # clip_root_path = f"./data/clips/SPdete-27/test/"
    # clip_root_path = f"./data/clips/SPgt-27/test/"
    # # clip_save_path = f"./temp_visualize/magfb-SPdete-27-test/"
    # # clip_save_path = f"./temp_visualize/magfb-SPgt-27-test/"
    # # clip_save_path = f"./temp_visualize/d3dp-SPdete-27-test/"
    # # clip_save_path = f"./temp_visualize/d3dp-SPgt-27-test/"
    # # clip_save_path = f"./temp_visualize/kasportsformer-SPdete-27-test/"
    # clip_save_path = f"./temp_visualize/kasportsformer-SPgt-27-test/"
    # # model_path = f"./checkpoints/evaluate_checkpoint/magfb.pth"
    # # model_path = f"./checkpoints/evaluate_checkpoint/d3dp.pth"
    # model_path = f"./checkpoints/evaluate_checkpoint/kasportsformer.pth"
    #
    # visual_clip_generate(args_dict=args_dict, clip_root_dir_path=clip_root_path, clip_save_dir_path=clip_save_path, model_path=model_path)

    # temp_clip_path = r"./temp_visualize/kasportsformer-SPdete-27-test/00000000.pkl"
    # plot_one_clip(temp_clip_path)

    # reference_clip_path_list = [r"./temp_visualize/d3dp-SPdete-27-test/", "./temp_visualize/kasportsformer-SPdete-27-test/"]
    # visualization_plot(r"./temp_visualize/magfb-SPdete-27-test/", r"./temp_visualize/plot/magfb-SPdete-27-test/", reference_clip_path_list)

    # reference_clip_path_list = [r"./temp_visualize/magfb-SPdete-27-test/", "./temp_visualize/kasportsformer-SPdete-27-test/"]
    # visualization_plot(r"./temp_visualize/d3dp-SPdete-27-test/", r"./temp_visualize/plot/d3dp-SPdete-27-test/", reference_clip_path_list)

    reference_clip_path_list = [r"./temp_visualize/magfb-SPdete-27-test/", "./temp_visualize/d3dp-SPdete-27-test/"]
    visualization_plot(r"./temp_visualize/kasportsformer-SPdete-27-test/", r"./temp_visualize/plot/kasportsformer-SPdete-27-test/", reference_clip_path_list)

    # compare_log_print(sports_clip_dir=r"./temp_visualize/kasportsformer-SPdete-27-test/", magf_clip_dir=r"./temp_visualize/magfb-SPdete-27-test/", d3dp_clip_dir=r"./temp_visualize/d3dp-SPdete-27-test/")





if __name__ == '__main__':
    main()




# def visualization_gif_plot(clip_root_dir: str, save_dir: str, interval: int) -> None:
#     print("gif generating")
#     clip_list = []
#     clip_file_path_list = sorted(os.listdir(clip_root_dir))
#     for i in clip_file_path_list:
#         clip_list.append(os.path.join(clip_root_dir, i))
#
#     for clip_index in tqdm(range(len(clip_list))):
#         with open(clip_list[clip_index], "rb") as f:
#             clip_data = pickle.load(f)
#
#         plot_predict = clip_data["predicted_result"]
#         plot_gt = clip_data["ground_truth"]
#         cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
#         plot_predict = plot_predict @ cam2real
#         plot_gt = plot_gt @ cam2real
#
#         gif_plot_generate_single(plot_predict, plot_gt, clip_index_str=clip_file_path_list[clip_index].split(".")[0], interval=interval, save_dir=save_dir)
#
#         # print(f"generated ! {clip_list[clip_index]}")
#         # print("eraly stop, for testing")
#         # exit()
#
#
#
# def gif_plot_generate_single(plot_predict, plot_gt, clip_index_str, interval, save_dir):
#     def update(frame):
#         ax.clear()
#         ax.set_xlim3d([min_value[0], max_value[0]])
#         ax.set_ylim3d([min_value[1], max_value[1]])
#         ax.set_zlim3d([min_value[2], max_value[2]])
#
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#
#         x_predict = plot_predict[frame, :, 0]
#         y_predict = plot_predict[frame, :, 1]
#         z_predict = plot_predict[frame, :, 2]
#
#         x_gt = plot_gt[frame, :, 0]
#         y_gt = plot_gt[frame, :, 1]
#         z_gt = plot_gt[frame, :, 2]
#
#         for connection in CONNECTIONS:
#             start = plot_predict[frame, connection[0], :]
#             end = plot_predict[frame, connection[1], :]
#             xs = [start[0], end[0]]
#             ys = [start[1], end[1]]
#             zs = [start[2], end[2]]
#             ax.plot(xs, ys, zs, c='skyblue')
#         for connection in CONNECTIONS:
#             start = plot_gt[frame, connection[0], :]
#             end = plot_gt[frame, connection[1], :]
#             xs = [start[0], end[0]]
#             ys = [start[1], end[1]]
#             zs = [start[2], end[2]]
#             ax.plot(xs, ys, zs, c='gray')
#
#         ax.scatter(x_predict, y_predict, z_predict, c='skyblue')
#         ax.scatter(x_gt, y_gt, z_gt, c='gray')
#
#         ax.view_init(elev=20, azim=60)
#
#         return ax
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     min_value_predict = np.min(plot_predict, axis=(0, 1))
#     max_value_predict = np.max(plot_predict, axis=(0, 1))
#     min_value_gt = np.min(plot_gt, axis=(0, 1))
#     max_value_gt = np.max(plot_gt, axis=(0, 1))
#
#     max_value = np.maximum(max_value_predict, max_value_gt)
#     min_value = np.minimum(min_value_predict, min_value_gt)
#
#     ani = FuncAnimation(fig, update, frames=plot_predict.shape[0], interval=interval)
#     # save_gif_file_name = clip_index_str + ".gif"
#     ani.save(save_dir + clip_index_str + ".gif")
#     plt.close()



# def dataset_reader(sequence_index: int, read_path: str = r'../saved_clips/SportsPose-27/test/'):
#     # cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
#     # scale_factor = 0.298
#
#     clip_path = read_path + '%06d.pkl' % sequence_index
#     clip_file = open(clip_path, 'rb')
#     file_content = pickle.load(clip_file)
#     clip_file.close()
#     sample_joint_seq = file_content['data_label']
#
#     # sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)
#     # sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
#
#     return sample_joint_seq
#
#
#
# def generate_args() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--sequence-index', type=int, default=10)
#     parser.add_argument('--dataset', type=str, default='SportsPose')
#     args = parser.parse_args()
#
#     print(f"Visualizing SportsPose Clips (sequence index: {args.sequence_index})")
#
#     def update(frame):
#         ax.clear()
#         ax.set_xlim3d([min_value[0], max_value[0]])
#         ax.set_ylim3d([min_value[1], max_value[1]])
#         ax.set_zlim3d([min_value[2], max_value[2]])
#
#         x = sample_joint_seq[frame, :, 0]
#         y = sample_joint_seq[frame, :, 1]
#         z = sample_joint_seq[frame, :, 2]
#
#         for connection in CONNECTIONS:
#             start = sample_joint_seq[frame, connection[0], :]
#             end = sample_joint_seq[frame, connection[1], :]
#             xs = [start[0], end[0]]
#             ys = [start[1], end[1]]
#             zs = [start[2], end[2]]
#             ax.plot(xs, ys, zs, c='red')
#
#         ax.scatter(x, y, z)
#
#         return ax
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     sample_joint_seq = dataset_reader(sequence_index=args.sequence_index)
#     print(f"Number of frames: {sample_joint_seq.shape[0]}")
#
#     min_value = np.min(sample_joint_seq, axis=(0, 1))
#     max_value = np.max(sample_joint_seq, axis=(0, 1))
#
#     ani = FuncAnimation(fig, update, frames=sample_joint_seq.shape[0], interval=20)
#     parent_dir = '../saved_clips/visualize/'
#     if not os.path.exists(parent_dir):
#         os.makedirs(parent_dir)
#     ani.save(f'{parent_dir}/SportsPose_27_{args.sequence_index}.gif')
#
# def repeat_generate(sequence_index: int, interval: int, read_path: str, save_dir: str):
#     def update(frame):
#         ax.clear()
#         ax.set_xlim3d([min_value[0], max_value[0]])
#         ax.set_ylim3d([min_value[1], max_value[1]])
#         ax.set_zlim3d([min_value[2], max_value[2]])
#
#         x = sample_joint_seq[frame, :, 0]
#         y = sample_joint_seq[frame, :, 1]
#         z = sample_joint_seq[frame, :, 2]
#
#         for connection in CONNECTIONS:
#             start = sample_joint_seq[frame, connection[0], :]
#             end = sample_joint_seq[frame, connection[1], :]
#             xs = [start[0], end[0]]
#             ys = [start[1], end[1]]
#             zs = [start[2], end[2]]
#             ax.plot(xs, ys, zs, c='green')
#
#         ax.scatter(x, y, z, c='green')
#         # ax.scatter(x, y, z, c='gray')
#
#         return ax
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     sample_joint_seq = dataset_reader(sequence_index=sequence_index, read_path=read_path)
#     # print(f"Number of frames: {sample_joint_seq.shape[0]}")
#
#     min_value = np.min(sample_joint_seq, axis=(0, 1))
#     max_value = np.max(sample_joint_seq, axis=(0, 1))
#
#     ani = FuncAnimation(fig, update, frames=sample_joint_seq.shape[0], interval=interval)
#     parent_dir = save_dir
#     if not os.path.exists(parent_dir):
#         os.makedirs(parent_dir)
#     ani.save(f'{parent_dir}/{sequence_index}.gif')
