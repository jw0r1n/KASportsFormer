import numpy as np
import torch
import copy
import glob
import cv2
from tqdm import tqdm
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.model_tools import load_model
from utils.utilities import yaml_config_reader
from lib.hrnet.gen_kpts import gen_video_kpts
from lib.preprocess import h36m_coco_format
from lib.utils import normalize_screen_coordinates, flip_data, camera_to_world


# plt.switch_backend('agg')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42




HUMAN36M_KINEMATIC_TREE = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
     [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
     [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

HUMAN36M_BGR_LINECOLORS = [
    (185, 128, 46),
    (14, 127, 255),
    (65, 169, 65),
    (62, 61, 218),
    (193, 113, 155),
    (84, 95, 146),
    (195, 121, 227),
    (129, 129, 129),
    (39, 191, 190),
    (209, 193, 35),
    (126, 221, 251),
    (208, 224, 64),
    (65, 15, 88),
    (238, 130, 238),
    (192, 129, 255),
    (38, 64, 239),
]


HUMAN36M_HEX_LINECOLORS = [
    "#2e80b9",
    "#ff7f0e",
    "#41a941",
    "#da3d3e",
    "#9b71c1",
    "#925f54",
    "#e379c3",
    "#818181",
    "#bebf27",
    "#23c1d1",
    "#fbdd7e",
    "#40e0d0",
    "#580f41",
    "#ee82ee",
    "#ff81c0",
    "#ef4026",
]



def detect_2d_pose(video_path, output_dir) -> None:
    print("\n🚀 2D Pose Detecting...")

    keypoints, scores = gen_video_kpts(video_path, det_dim=416, num_person=1)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir = os.path.join(output_dir, 'detect2d')
    os.makedirs(output_dir, exist_ok=True)

    output_2D_path = os.path.join(output_dir, 'keypoints2d.pkl')

    with open(output_2D_path, 'wb') as f:
        pickle.dump(keypoints, f)




def plot_on_frame(kpts, img):
    connections = HUMAN36M_KINEMATIC_TREE
    line_colors = HUMAN36M_BGR_LINECOLORS
    dot_color = (0, 255, 255)
    thickness = 2
    for j, c in enumerate(connections):
        start = list(map(int, kpts[c[0]]))
        end = list(map(int, kpts[c[1]]))
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), line_colors[j], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=dot_color, radius=2)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=dot_color, radius=2)

    return img



def plot_2d_pose(video_path, output_dir) -> None:
    print("\n🚀 Plotting 2D pose...")
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir_2D = os.path.join(output_dir, 'pose2d')
    os.makedirs(output_dir_2D, exist_ok=True)

    with open(os.path.join(output_dir, 'detect2d', 'keypoints2d.pkl'), "rb") as f:
        keypoints = pickle.load(f)
    keypoints = keypoints[0]

    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        # img_size = img.shape # (720, 1280, 3)
        if img is None:
            continue
        input_2D = keypoints[i]
        image = plot_on_frame(input_2D, copy.deepcopy(img))
        cv2.imwrite(os.path.join(output_dir_2D, '%04d' % i + '_2D.png'), image)



def resample(n_frames, target_frame):
    even = np.linspace(0, n_frames, num=target_frame, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints, target_frame_length):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= target_frame_length:
        new_indices = resample(n_frames, target_frame_length)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, target_frame_length):
            keypoints_clip = keypoints[:, start_idx:start_idx + target_frame_length, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != target_frame_length:
                new_indices = resample(clip_length, target_frame_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample


def plot_3d(vals, ax, elev=20, azim=10):
    ax.view_init(elev=elev, azim=azim)

    x_predict = vals[:, 0]
    y_predict = vals[:, 1]
    z_predict = vals[:, 2]
    for i, connection in enumerate(HUMAN36M_KINEMATIC_TREE):
        start =  vals[connection[0], :]
        end = vals[connection[1], :]
        xs = [start[0], end[0]]
        ys = [start[1], end[1]]
        zs = [start[2], end[2]]
        ax.plot(xs, ys, zs, c=HUMAN36M_HEX_LINECOLORS[i])

    ax.scatter(x_predict, y_predict, z_predict, c='yellow')

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def lift_3d_pose(video_path, output_dir, config_path, model_path, elev=0, azim=0) -> None:
    print("\n🚀 Generating 3D Pose...")

    cap = cv2.VideoCapture(video_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 720
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1280


    args_dict = yaml_config_reader(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'    current device: {device}')
    model = load_model(args_dict)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"    model {args_dict.model_name} checkpoint loaded")
    else:
        raise Exception('check your dumbass checkpoint path')
    model.eval()

    with open(os.path.join(output_dir, 'detect2d', 'keypoints2d.pkl'), "rb") as f:
        keypoints = pickle.load(f)

    clips, downsample = turn_into_clips(keypoints, 27)

    output_dir_3D = os.path.join(output_dir, 'pose3d')
    os.makedirs(output_dir_3D, exist_ok=True)

    for idx, clip in tqdm(enumerate(clips)):
        input_2D = normalize_screen_coordinates(clip, w=frame_width, h=frame_height)
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        for j, post_out in enumerate(post_out_all):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            plot_3d(post_out, ax, elev, azim)

            save_path = os.path.join(output_dir_3D, '%04d' % (idx * 27 + j) + '_3D.png')
            plt.savefig(save_path, dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)


def plot_image(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)

def demo_figure_generate(output_dir) -> None:
    print("\n🚀 Generating Demo Figure...")

    image_2d_dir = sorted(glob.glob(os.path.join(output_dir, 'pose2d', '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir, 'pose3d', '*.png')))

    output_dir_pose = os.path.join(output_dir, 'demo')
    os.makedirs(output_dir_pose, exist_ok=True)

    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        if image_2d.shape[0] > image_2d.shape[1]:
            edge2d = (image_2d.shape[0] - image_2d.shape[1]) // 2
            image_2d = image_2d[edge2d:image_2d.shape[0] - edge2d, :]
        else:
            edge2d = (image_2d.shape[1] - image_2d.shape[0]) // 2
            image_2d = image_2d[:, edge2d:image_2d.shape[1] - edge2d]

        edge3d = 130
        image_3d = image_3d[edge3d:image_3d.shape[0] - edge3d, edge3d:image_3d.shape[1] - edge3d]

        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        plot_image(ax, image_2d)
        # ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        plot_image(ax, image_3d)
        # ax.set_title("Reconstruction", fontsize = font_size)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path = os.path.join(output_dir_pose, '%04d' % i + '_demo.png')
        plt.savefig(save_path, dpi=200, bbox_inches = 'tight')
        plt.close(fig)


def demo_video_generate(video_path, output_dir):
    print("\n🚀 Generating Demo Video...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    demo_files = sorted(glob.glob(os.path.join(output_dir, 'demo', '*.png')))
    temp_img = cv2.imread(demo_files[0])
    size = (temp_img.shape[1], temp_img.shape[0])

    video_write = cv2.VideoWriter(os.path.join(output_dir, 'demo.mp4'), fourcc, fps, size)

    for frame in tqdm(demo_files):
        img = cv2.imread(frame)
        video_write.write(img)

    video_write.release()


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video3.mp4', help='input video')
    parser.add_argument('--config', type=str, default=r"../configs/sportspose-kasportsformer.yaml", help='configuration file path')
    parser.add_argument('--model', type=str, default=r"../checkpoints/evaluate_checkpoint/kasportsformer-sp-gt.pth", help='checkpoint file path')
    parser.add_argument('--elev', type=int, default=5, help='elev')
    parser.add_argument('--azim', type=int, default=5, help='azim')
    args = parser.parse_args()

    video_path = os.path.join('./video', args.video)
    config_path = args.config
    model_path = args.model
    elev = int(args.elev)
    azim = int(args.azim)

    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join('./output', video_name)

    detect_2d_pose(video_path, output_dir)
    plot_2d_pose(video_path, output_dir)
    lift_3d_pose(video_path, output_dir, config_path, model_path, elev=elev, azim=azim)
    demo_figure_generate(output_dir)
    demo_video_generate(video_path, output_dir)





if __name__ == '__main__':
    main()

