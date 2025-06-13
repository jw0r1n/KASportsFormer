import argparse
import os

import numpy as np
import torch

import wandb
from easydict import EasyDict
from tqdm import tqdm

from utils.utilities import (yaml_config_reader, same_seed_fix, get_logger,
                             AverageMetering, checkpoint_save, decay_learning_rate_exponentially, joint_flip)
from utils.static_values import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS
from utils.loss_calc import (mpjpe_loss_calc, n_mpjpe_loss_calc, velocity_loss_calc, loss_limb_var_calc,
                             loss_limb_len_calc, loss_cos_simi_calc, loss_cos_simi_velocity_calc, weighted_mpjpe, mean_velocity_error_train)
from utils.error_calc import mpjpe_calc, jpe_calc, acc_error_calc, p_mpjpe_calc
from torch.utils.data import DataLoader
from torch import optim
from wandb import util
from model.model_tools import load_model, total_parameters_count

from data.reader.sp_dataset import SportsPose3DDataset
from data.reader.sp_reader import DataReaderSportsPose



def evaluate_one_epoch_new(args_dict, model, test_loader, device, epoch_index, logwriter):
    # WARNING: No Test frame overlay is allowed in this version

    model.eval()

    result_per_activity = {}
    result_procrustes_per_activity = {}
    result_joints_per_activity = [{} for _ in range(args_dict.num_joints)]
    result_acceleration_per_activity = {}

    action_name_set = set()

    with torch.no_grad():
        for joint_input, joint_label_scaled, joint_factor, joint_action, joint_res in tqdm(test_loader,
                                                                                           desc=f'Evaluating Epoch... ({epoch_index + 1})' if not args_dict.eval_only else 'Evaluation Only Running...'):
            joint_input = joint_input.to(device)  # (8, 27, 17, 3)
            # joint_label_scaled = joint_label_scaled.to(device) # (8, 27, 17, 3)
            # joint_factor = joint_factor.to(device) # (8, 27)
            # joint_res = joint_res.to(device) # (8, 2)
            if args_dict.flip:
                batch_input_flip = joint_flip(joint_input)
                predicted_pos_original = model(joint_input)
                predicted_pos_flip = model(batch_input_flip)
                predicted_pos_flip = joint_flip(predicted_pos_flip)
                predicted_result = (predicted_pos_original + predicted_pos_flip) / 2
            else:
                predicted_result = model(joint_input)

            predicted_result[:, :, 0, :] = 0  # (8, 27, 17, 3)

            predicted_result = predicted_result.cpu().numpy()
            joint_label_scaled = joint_label_scaled.cpu().numpy()
            joint_factor = joint_factor.cpu().numpy()
            joint_res = joint_res.cpu().numpy()

            for idx, predicted_data in enumerate(predicted_result):  # de-normalization
                res_w, res_h = joint_res[idx]
                # frame_indexes = frame_index_clips[idx]
                predicted_data[:, :, :2] = (predicted_data[:, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
                predicted_data[:, :, 2:] = predicted_data[:, :, 2:] * res_w / 2

                factor = joint_factor[idx][:, None, None]
                ground_truth = joint_label_scaled[idx]
                predicted_data *= factor
                predicted_data = predicted_data - predicted_data[:, 0:1, :]
                ground_truth = ground_truth - ground_truth[:, 0:1, :]

                mpjpe = mpjpe_calc(predicted_data, ground_truth) # (27, )
                # mpjpe_all[frame_indexes] += mpjpe
                jpe = jpe_calc(predicted_data, ground_truth) # (27, 17, )
                # for joint_idx in range(args_dict.num_joints):
                #     jpe_all_joints[frame_indexes, joint_idx] += jpe[:, joint_idx]
                acc_err = acc_error_calc(predicted_data, ground_truth)
                # acc_err_all[frame_indexes[:-2]] += acc_err
                p_mpjpe = p_mpjpe_calc(predicted_data, ground_truth)
                # p_mpjpe_all[frame_indexes] += p_mpjpe
                # overlay_count[frame_indexes] += 1

                current_action = joint_action[idx]
                action_name_set.add(current_action)

                if current_action not in result_per_activity:
                    result_per_activity[current_action] = []
                result_per_activity[current_action].extend(mpjpe)

                if current_action not in result_procrustes_per_activity:
                    result_procrustes_per_activity[current_action] = []
                result_procrustes_per_activity[current_action].extend(p_mpjpe)

                if current_action not in result_acceleration_per_activity:
                    result_acceleration_per_activity[current_action] = []
                result_acceleration_per_activity[current_action].extend(acc_err)

                for joint_idx in range(args_dict.num_joints):
                    if current_action not in result_joints_per_activity[joint_idx]:
                        result_joints_per_activity[joint_idx][current_action] = []
                    result_joints_per_activity[joint_idx][current_action].extend(jpe[:, joint_idx])

    final_result_mpjpe_per_action = []
    final_result_procrustes_per_action = []
    final_result_acceleration_per_action = []
    final_result_joints_per_action = [[] for _ in range(args_dict.num_joints)]

    action_name_sequence = []
    for action_name in action_name_set:
        action_name_sequence.append(action_name)
        final_result_mpjpe_per_action.append(np.mean(result_per_activity[action_name]))
        final_result_procrustes_per_action.append(np.mean(result_procrustes_per_activity[action_name]))
        final_result_acceleration_per_action.append(np.mean(result_acceleration_per_activity[action_name]))
        for joint_idx in range(args_dict.num_joints):
            final_result_joints_per_action[joint_idx].append(np.mean(result_joints_per_activity[joint_idx][action_name]))

    joint_errors_count_for_all = []
    for joint_idx in range(args_dict.num_joints):
        joint_errors_count_for_all.append(np.mean(np.array(final_result_joints_per_action[joint_idx])))
    joint_errors_count_for_all = np.array(joint_errors_count_for_all)

    error1 = np.mean(np.array(final_result_mpjpe_per_action))
    # assert round(e1, 4) == round(np.mean(joint_errors_count_for_all), 4), f"MPJPE{e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors_count_for_all):.4f}"
    acceleration_error = np.mean(np.array(final_result_acceleration_per_action))
    error2 = np.mean(np.array(final_result_procrustes_per_action))

    evaluate_result_dict = {}
    evaluate_result_dict["mpjpe"] = error1
    evaluate_result_dict["p_mpjpe"] = error2
    evaluate_result_dict["acceleration_error"] = acceleration_error
    evaluate_result_dict["activity_name_sequence"] = action_name_sequence
    evaluate_result_dict["mpjpe_activity"] = final_result_mpjpe_per_action
    evaluate_result_dict["mpjpe_joint"] = joint_errors_count_for_all


    action_result_log_message = ""
    for action_index, action_name in enumerate(action_name_sequence):
        action_result_log_message = action_result_log_message + "\n" + action_name + ": " + str(final_result_mpjpe_per_action[action_index])
    logwriter.info(action_result_log_message)


    joint_result_log_message = ""
    for joint_idx in range(args_dict.num_joints):
        joint_result_log_message = joint_result_log_message + "\n" + "joint_idx: " + str(joint_idx) + " " + H36M_JOINT_TO_LABEL[joint_idx] + " " + str(joint_errors_count_for_all[joint_idx])
    logwriter.info(joint_result_log_message)

    return evaluate_result_dict


def evaluate(args_dict: EasyDict) -> None:

    logwriter = get_logger(args_dict.logger_dir_path, f'{args_dict.wandb_name}_evaluate.log')
    logwriter.info("Start Evaluating...")

    test_dataset = SportsPose3DDataset(args_dict=args_dict, data_split='test')
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args_dict.batch_size, num_workers=args_dict.num_cpus - 1,
                             pin_memory=args_dict.pin_memory, prefetch_factor=args_dict.num_cpus // 3,
                             persistent_workers=args_dict.persistent_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args_dict)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    params_count = total_parameters_count(model)
    logwriter.info(f'The total parameter numbers of this model: {params_count:,}')

    checkpoint_path = os.path.join(args_dict.evaluate_checkpoint_file_dir, args_dict.evaluate_checkpoint_file)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        raise Exception('evaluation checkpoint is wrong, check your configuration')

    evaluate_result = evaluate_one_epoch_new(args_dict, model, test_loader, device, -1, logwriter)

    mpjpe, p_mpjpe, acceleration_error, mpjpe_per_activity, activity_name_sequence, mpjpe_per_joint = \
        evaluate_result["mpjpe"], evaluate_result["p_mpjpe"], evaluate_result["acceleration_error"], evaluate_result[
            "mpjpe_activity"], evaluate_result["activity_name_sequence"], evaluate_result["mpjpe_joint"]

    joint_errors_with_label = {}
    for joint_idx in range(args_dict.num_joints):
        joint_errors_with_label[f"{H36M_JOINT_TO_LABEL[joint_idx]}"] = mpjpe_per_joint[joint_idx]


    main_result_log_message = '\n' + f"Protocol #1 Error (MPJPE): {mpjpe} mm" + '\n' + f"Protocol #2 Error (P_MPJPE): {p_mpjpe} mm" + '\n' + f"(Acceleration Error) {acceleration_error} mm^2"
    logwriter.info(main_result_log_message)

    upperbody_log_message = "\n" + f'(Upper Body Joint MPJPE: {np.mean(mpjpe_per_joint[H36M_UPPER_BODY_JOINTS])})'
    for i in H36M_UPPER_BODY_JOINTS:
        upperbody_log_message = upperbody_log_message + "\n" + H36M_JOINT_TO_LABEL[i] + str(mpjpe_per_joint[i])
    logwriter.info(upperbody_log_message)
    lowerbody_log_message = "\n" + f'(Lower Body Joint MPJPE: {np.mean(mpjpe_per_joint[H36M_LOWER_BODY_JOINTS])})'
    for i in H36M_LOWER_BODY_JOINTS:
        lowerbody_log_message = lowerbody_log_message + "\n" + H36M_JOINT_TO_LABEL[i] + str(mpjpe_per_joint[i])
    logwriter.info(lowerbody_log_message)

def train_one_epoch(args_dict, model, train_loader: DataLoader, optimizer, device, epoch_losses_dict, epoch_index) -> None:
    model.train()
    train_progress_bar = tqdm(train_loader, desc=f'Training Epoch... ({epoch_index + 1})')
    for x, y in train_progress_bar:
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        predict_result = model(x)

        optimizer.zero_grad()

        loss_mpjpe = mpjpe_loss_calc(predict_result, y)
        loss_n_mpjpe = n_mpjpe_loss_calc(predict_result, y)
        loss_velocity = velocity_loss_calc(predict_result, y)

        # loss_limb_len_var = loss_limb_var_calc(predict_result)
        # loss_limb_len = loss_limb_len_calc(predict_result, y)
        # loss_limb_len_cos_simi = loss_cos_simi_calc(predict_result, y)
        # loss_limb_len_cos_simi_velocity = loss_cos_simi_velocity_calc(predict_result, y)
        # loss_total_complete = loss_mpjpe + args_dict.lambda_n_mpjpe * loss_n_mpjpe + args_dict.lambda_mpjpe_velocity * loss_velocity + args_dict.lambda_limb_len_var * loss_limb_len_var + args_dict.lambda_limb_len * loss_limb_len + args_dict.lambda_limb_cos_simi * loss_limb_len_cos_simi + args_dict.lambda_limb_cos_simi_velocity * loss_limb_len_cos_simi_velocity

        loss_total = loss_mpjpe + args_dict.lambda_n_mpjpe * loss_n_mpjpe + args_dict.lambda_mpjpe_velocity * loss_velocity

        epoch_losses_dict['loss_mpjpe'].update(loss_mpjpe.item(), batch_size)
        epoch_losses_dict['loss_n_mpjpe'].update(loss_n_mpjpe.item(), batch_size)
        epoch_losses_dict['loss_velocity'].update(loss_velocity.item(), batch_size)

        # epoch_losses_dict['loss_limb_len_var'].update(loss_limb_len_var.item(), batch_size)
        # epoch_losses_dict['loss_limb_len'].update(loss_limb_len.item(), batch_size)
        # epoch_losses_dict['loss_limb_len_cos_simi'].update(loss_limb_len_cos_simi.item(), batch_size)
        # epoch_losses_dict['loss_limb_len_cos_simi_velocity'].update(loss_limb_len_cos_simi_velocity.item(), batch_size)

        epoch_losses_dict['loss_total'].update(loss_total.item(), batch_size)

        train_progress_bar.set_postfix({
            'loss_mpjpe': epoch_losses_dict['loss_mpjpe'].avg,
            'loss_n_mpjpe': epoch_losses_dict['loss_n_mpjpe'].avg,
            'loss_total': epoch_losses_dict['loss_total'].avg
        })

        loss_total.backward()

        optimizer.step()


def train(args_dict: EasyDict) -> None:

    train_dataset = SportsPose3DDataset(args_dict=args_dict, data_split='train')
    test_dataset = SportsPose3DDataset(args_dict=args_dict, data_split='test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args_dict.batch_size, num_workers=args_dict.num_cpus - 1,
                              pin_memory=args_dict.pin_memory, prefetch_factor=args_dict.num_cpus // 3, persistent_workers=args_dict.persistent_workers)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args_dict.batch_size, num_workers=args_dict.num_cpus - 1,
                              pin_memory=args_dict.pin_memory, prefetch_factor=args_dict.num_cpus // 3,
                              persistent_workers=args_dict.persistent_workers)


    logwriter = get_logger(args_dict.logger_dir_path, args_dict.logger_file_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args_dict)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    params_count = total_parameters_count(model)
    logwriter.info(f'The total parameter numbers of this model: {params_count:,}')

    learning_rate = args_dict.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate,
                            weight_decay=args_dict.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args_dict.learning_rate_decay, patience=2)

    epoch_start = 0
    min_mpjpe = float('inf')

    # if args_dict.use_wandb:
    if 'wandb_run_id' in args_dict and args_dict.wandb_run_id != '':
        wandb_run_id = args_dict.wandb_run_id
    else:
        wandb_run_id = wandb.util.generate_id()
    logwriter.info(f'current wandb id: {wandb_run_id}')

    if args_dict.checkpoint:
        checkpoint_path = os.path.join(args_dict.checkpoint_dir, args_dict.checkpoint_file_name)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            logwriter.info(f"checkpoint loaded! ({checkpoint_path})")
            # model.load_state_dict(checkpoint['model'], strict=False)

            if args_dict.resume:
                learning_rate = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_run_id' in checkpoint and args_dict.wandb_run_id is None:
                    wandb_run_id = checkpoint['wandb_run_id']
        else:
            raise Exception('checkpoint path is wrong, check your configuration')

    # WandB initial setup:
    if not args_dict.eval_only and args_dict.use_wandb:
        if args_dict.resume:
            print(f'initializing wandb in resume mode...current wandb id: {wandb_run_id}')
            wandb.init(id=wandb_run_id, name=args_dict.wandb_name ,project=args_dict.wandb_project_name, resume='must', settings=wandb.Settings(start_method='fork'))
        else:
            print(f'initializing wandb...current wandb id: {wandb_run_id}')
            wandb.init(id=wandb_run_id, name=args_dict.wandb_name, project=args_dict.wandb_project_name, settings=wandb.Settings(start_method='fork'))
            wandb.config.update({'run_id': wandb_run_id})
            wandb.config.update(args_dict)
            # installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
            # wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(args_dict.new_checkpoint_dir, f'{args_dict.new_checkpoint_name}_epoch_latest.pth')
    checkpoint_path_best = os.path.join(args_dict.new_checkpoint_dir, f'{args_dict.new_checkpoint_name}_epoch_best.pth')

    training_patience_count = 0
    min_mpjpe_epoch_number = 0

    for epoch in range(epoch_start, args_dict.epochs):
        logwriter.info(f'train epoch: {epoch + 1} ...')

        if args_dict.warmup and epoch <= args_dict.warmup_epoches:
            warmup_start = args_dict.learning_rate / 100
            current_lr = warmup_start + (args_dict.learning_rate - warmup_start) * (epoch / args_dict.warmup_epoches)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        # loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        # epoch_losses_dict = {name: AverageMetering() for name in loss_names}
        # loss_record_names_complete = ['loss_mpjpe', 'loss_n_mpjpe', 'loss_velocity', 'loss_limb_len_var', 'loss_limb_len', 'loss_limb_len_cos_simi', 'loss_limb_len_cos_simi_velocity', 'loss_total']
        loss_record_names = ['loss_mpjpe', 'loss_n_mpjpe', 'loss_velocity', 'loss_total']
        epoch_losses_dict = {name: AverageMetering() for name in loss_record_names}

        train_one_epoch(args_dict=args_dict, model=model, train_loader=train_loader,
                        optimizer=optimizer, device=device, epoch_losses_dict=epoch_losses_dict, epoch_index=epoch)
        evaluate_result = evaluate_one_epoch_new(args_dict, model, test_loader, device, epoch, logwriter)


        mpjpe, p_mpjpe, acceleration_error, mpjpe_per_activity, activity_name_sequence, mpjpe_per_joint = \
        evaluate_result["mpjpe"], evaluate_result["p_mpjpe"], evaluate_result["acceleration_error"], evaluate_result[
            "mpjpe_activity"], evaluate_result["activity_name_sequence"], evaluate_result["mpjpe_joint"]

        logwriter.info(f'train epoch {epoch + 1} result: MPJPE {mpjpe} mm   P-MPJPE {p_mpjpe} mm   acceleration_error {acceleration_error} mm/s^2')

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            training_patience_count = 0
            min_mpjpe_epoch_number = epoch
            checkpoint_save(checkpoint_path_best, epoch, learning_rate, optimizer, model, mpjpe, wandb_run_id)
            logwriter.info(f"checkpoint saved at ({checkpoint_path_best}) with mpjpe ({mpjpe})")
        else:
            training_patience_count = training_patience_count + 1
        checkpoint_save(checkpoint_path_latest, epoch, learning_rate, optimizer, model, mpjpe, wandb_run_id)

        joint_label_errors = {}
        for joint_idx in range(args_dict.num_joints):
            joint_label_errors[f"eval_joint/{H36M_JOINT_TO_LABEL[joint_idx]}"] = mpjpe_per_joint[joint_idx]
        activity_errors = {}
        for idx, activity in enumerate(activity_name_sequence):
            activity_errors[f"eval_activity/{activity}"] = mpjpe_per_activity[idx]

        if args_dict.use_wandb:
            wandb.log({
                'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
                'train/loss_mpjpe': epoch_losses_dict['loss_mpjpe'].avg,
                'train/loss_n_mpjpe': epoch_losses_dict['loss_n_mpjpe'].avg,
                'train/loss_velocity': epoch_losses_dict['loss_velocity'].avg,
                # 'train/loss_limb_len_var': epoch_losses_dict['loss_limb_len_var'].avg,
                # 'train/loss_limb_len': epoch_losses_dict['loss_limb_len'].avg,
                # 'train/loss_limb_len_cos_simi': epoch_losses_dict['loss_limb_len_cos_simi'].avg,
                # 'train/loss_limb_len_cos_simi_velocity': epoch_losses_dict['loss_limb_len_cos_simi_velocity'].avg,
                'train/loss_total': epoch_losses_dict['loss_total'].avg,

                'eval/mpjpe': mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval/min_mpjpe': min_mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval_additional/upper_body_mpjpe': np.mean(mpjpe_per_joint[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_mpjpe': np.mean(mpjpe_per_joint[H36M_LOWER_BODY_JOINTS]),
                # 'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                # 'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                # 'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors,
                **activity_errors,
            }, step=epoch+1)


        if args_dict.warmup:
            if epoch > args_dict.warmup_epoches:
                scheduler.step(mpjpe)
        else:
            scheduler.step(mpjpe)

        if training_patience_count >= args_dict.training_epoch_patience:
            logwriter.info(f"Model is not improving for {training_patience_count} epoches, early stopping! 🫡")
            logwriter.info(f"Min MPJPE: {min_mpjpe} on epoch: {min_mpjpe_epoch_number + 1}")
            break


    # if args_dict.use_wandb:
    #     artifact = wandb.Artifact(f'model_file_save', type='model')
    #     artifact.add_file(checkpoint_path_latest)
    #     artifact.add_file(checkpoint_path_best)
    #     wandb.log_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="configs/sportspose-det-kasportsformer.yaml", help="path to the config file")
    cli_args = parser.parse_args()
    args_dict = yaml_config_reader(cli_args.config_path)
    same_seed_fix(args_dict.seed)
    if args_dict.use_wandb:
        os.environ["WANDB_API_KEY"] = args_dict.wandb_api_key
    if args_dict.eval_only:
        evaluate(args_dict)
    else:
        train(args_dict)


if __name__ == '__main__':
    main()

