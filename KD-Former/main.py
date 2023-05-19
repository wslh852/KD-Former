from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import kinematics
import kinematics_cmu
from human_36m import human_36m_dataset
from cmu_mocap import cmu_mocap_dataset
from batch_sample import generate_train_data, get_batch_srnn, get_batch_srnn_cmu
from seq2seq import EncodeDecodeModel
import torch_utils
from utils import Logger, AverageMeter
torch.backends.cudnn.enabled = False
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=0, help='use cpu only')
    parser.add_argument('--dataset', default='h36m', help='h36m or cmu')
    parser.add_argument('--data_dir', default='./h3.6m/dataset')
    parser.add_argument('--data_dir_cmu', default='./cmu_mocap/')
    parser.add_argument('--num_joint_cmu', type=int, default=38, help='number of joints for cmu dataset')
    parser.add_argument('--log_file', default='log_train.txt')
    parser.add_argument('--save_path', default='./temp/')

    parser.add_argument('--source_seq_len', type=int, default=50, help='length of encode sequence')
    parser.add_argument('--target_seq_len', type=int, default=25, help='length of output decode sequence')
    parser.add_argument('--num_joint', type=int, default=32, help='input size at each timestep')
    parser.add_argument('--in_chans', type=int, default=3, help='number of input channels')
    parser.add_argument('--embed_dim_ratio', type=int, default=8, help='hidden size of RNN')
    parser.add_argument('--depth', type=int, default=4, help='number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--qkv_bias', default=True, help='enable bias for qkv if True')
    parser.add_argument('--loss_type', type=int, default=1, help='0:expmap, 1:quaternion')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--qk_scale', type=float, default=None, help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--drop_rate',type=float , default=0.2, help='dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.2, help='attention dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='stochastic depth rate') 
   
    

    args = parser.parse_args()
    args.log_dir = 'log/' + args.dataset
    sys.stdout = Logger(os.path.join(args.log_dir, args.log_file))
    print(args)

    if args.dataset == 'h36m':
        dataset = human_36m_dataset
    if args.dataset == 'cmu':
        dataset = cmu_mocap_dataset
        args.data_dir = args.data_dir_cmu
        args.num_joint = args.num_joint_cmu

    model = EncodeDecodeModel(args.source_seq_len, args.target_seq_len, args.num_joint, args.in_chans, args.embed_dim_ratio, args.depth,
                              args.num_heads, args.qkv_bias, args.loss_type, args.mlp_ratio, args.qk_scale, args.drop_rate, args.attn_drop_rate,
                              args.drop_path_rate)

    dataset = dataset(args.data_dir)
   
    test_set,completeData = dataset.load_data(dataset.get_test_subject_ids())

    
       # if epoch_i ==1:
    total_err_lst = []
    print("{0: <18} |".format("milliseconds"), end="")
    for ms in [80, 160, 320, 400, 560, 1000]:
                print(" {0:5d} |".format(ms), end="")
    print()
    error_all=[]
    for action, action_idx in dataset.get_test_actions():
                error_list=[]
                # Evaluate the model on the test batches
                if args.dataset == 'h36m':
                    source_tst, target_tst = get_batch_srnn(test_set, action, args.source_seq_len, args.target_seq_len, 195)
                else:
                    source_tst, target_tst = get_batch_srnn_cmu(test_set, action, args.source_seq_len, args.target_seq_len, 3 * args.num_joint+3)

                source_tst = torch.tensor(source_tst).cuda()
                target_tst = torch.tensor(target_tst).cuda()

                # Discard the first joint, which represents a corrupted translation
                source_tst = source_tst[:, :, 3:].float()
                target_tst = target_tst[:, :, 3:].float()
                pred_target = model.eval(source_tst, target_tst)

                target_tst = torch_utils.tensor_expmap_to_euler(target_tst)

                # Convert from exponential map to Euler angles
                pred_target = torch_utils.tensor_expmap_to_euler(pred_target)

     
                error = torch.pow(target_tst[:,:,:96] - pred_target, 2)
                error = torch.sqrt(torch.sum(error, dim=-1) )
                error = torch.mean(error, dim=0)
                mean_mean_errors = error.cpu().detach().numpy()
                total_err_lst.append(np.mean(mean_mean_errors) )

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <18} |".format(action), end="")
                for ms in [1, 3, 7, 9, 13, 24]:
                    if mean_mean_errors.shape[0] >= ms + 1:
                        print(" {0:.2f} |".format(mean_mean_errors[ms]), end="")
                        error_list.append(mean_mean_errors[ms])
                    else:
                        print("   n/a |", end="")
                print() # start new line
                error_all.append(error_list)
    
    total_err = np.mean(total_err_lst)
    
if __name__ == '__main__':
    main()