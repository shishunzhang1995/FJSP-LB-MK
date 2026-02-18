import copy
import json
import os
import random
import time
from collections import deque
from env.case_generator import CaseGenerator
import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
import PPO_model
import PPO_model_ope_overlap
import PPO_model_train_file
import PPO_model_no_matrix_similarity
import PPO_model_box,PPO_model_box_MHA_2
# import PPO_model_dag_train
# from env_linear_ope.case_generator import CaseGenerator
from validate_LB import validate, get_validate_env
from torch.utils.tensorboard import SummaryWriter
from utils.my_utils import *
from env.load_data import nums_extract

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    scene_name="恒力利材线"
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    writer = SummaryWriter("logs/train_FJSP_LB/fenjian_part_switch_box_fenjian_edge_weight_sigmoid_3_proc_time_max_LB_similarity_1_10_switch"
                           "_reward_change_5_1e-3_lr_batch_20_{0}".format(str_time))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("config_train_file.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    # num_jobs = env_paras["num_jobs"]  #10
    # num_mas = env_paras["num_mas"]    #5
    # opes_per_job_min = int(num_mas * 0.8)  ##4
    # opes_per_job_max = int(num_mas * 1.2)  ##6
    memories = PPO_model_box.Memory()
    model = PPO_model_box.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"]) ##20
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    # if is_viz:
    #     viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save_train_file/train_{0}'.format(str_time)
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    valid_results = []
    valid_results_100 = []
    ave_switch=[]
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    writer_ave.save()
    writer_ave.close()
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    writer_100.save()
    writer_100.close()

    # Start training iteration
    start_time = time.time()
    env = None
    config_file = '/home/zss/data/00_FJSP/25_0502_FJSP_LB/dataset' \
                  '/DataWarehouse-master/数据集/plate_part/' + scene_name + '_layout.json'
    layout = read_json(config_file)
    for i in range(1, train_paras["max_iterations"]+1):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            # file_path = "./dataset/DataWarehouse-master/数据集/train_set/"+scene_name+"/"
            file_path ="/home/zss/data/00_FJSP/1_hgnn_jiexi/data_dev/1005/"
            # file_path = "./dataset/DataWarehouse-master/数据集/plate_part/"+scene_name+"/"
            train_data_files = os.listdir(file_path)
            picknumber = env_paras["batch_size"]  # 从文件夹中取一定数量的文件 20
            #
            # # picknumber = 2
            sample = random.sample(train_data_files, picknumber)
            # print("sample=",sample)
            for sample_index in range(len(sample)):
                sample[sample_index] = file_path + sample[sample_index]
                # sample[i] = file_path + '10j_5m_001.fjs'
            # print("sample=",sample)
            num_jobs=env_paras["num_jobs"]
            num_mas = env_paras["num_mas"]
            opes_per_job_min = int(num_mas * 0.8)  ##4
            opes_per_job_max = int(num_mas * 1.2)  ##6
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras,layout=config_file,mode='Train') ##batch 20
            # print('num_job: ', num_jobs, '\tnum_mas: ', num_mas)

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        model_predict_time = 0
        while ~done:
            with torch.no_grad():
                start_model_predict = time.time()
                actions = model.policy_old.act(state, memories, dones)
                end_model_predict = time.time()
                duration = end_model_predict - start_model_predict
                model_predict_time += duration
            start_step = time.time()
            state, rewards, dones = env.step(actions)
            # print("step time=", time.time() - start_step)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("model predict time=", model_predict_time)
        print("spend_time: ", time.time()-last_time)
        # Verify the solution
        # gantt_result = env.validate_gantt()[0]
        # if not gantt_result:
        #     print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            writer.add_scalar('Train/Loss', loss, i)
            writer.add_scalar('Train/Reward', reward, i)
            memories.clear_memory()
            # if is_viz:
            #     viz.line(X=np.array([i]), Y=np.array([reward]),
            #         win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
            #     viz.line(X=np.array([i]), Y=np.array([loss]),
            #         win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100,ave_switch_times = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            ave_switch.append(ave_switch_times)
            valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}.pt'.format(save_path,i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)
            writer.add_scalar('Train/Makespan of valid', vali_result.item(), i)
            writer.add_scalar('Train/Switch times of valid', ave_switch_times, i)
            # if is_viz:
            #     viz.line(
            #         X=np.array([i]), Y=np.array([vali_result.item()]),
            #         win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

    # Save the data of training curve to files
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave.save()
    writer_ave.close()
    column = [i_col for i_col in range(100)]
    data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
    writer_100.save()
    writer_100.close()
    writer.close()

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()