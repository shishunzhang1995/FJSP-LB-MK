import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator, CaseReader
from validate import validate, get_validate_env


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]      # 环境参数
    model_paras = load_dict["model_paras"]  # 模型参数
    train_paras = load_dict["train_paras"]  # 训练参数
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    env_valid = get_validate_env(env_valid_paras)  # 创建验证环境
    maxlen = 1  
    best_models = deque()
    makespan_best = float('inf')

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])

    with pd.ExcelWriter(f'{save_path}/training_ave_{str_time}.xlsx') as writer_ave:
        data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter(f'{save_path}/training_100_{str_time}.xlsx') as writer_100:
        data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    valid_results = []
    valid_results_100 = []

    # Start training iteration
    start_time = time.time()
    env = None
    root_dir = f"项目数据集/训练集"
    sub_dirs = os.listdir(root_dir)  # 子目录列表

    current_sub_dir_index = 0  # 当前子目录索引
    current_iteration = 0  # 当前迭代次数
    num_instances = len(os.listdir(os.path.join(root_dir, sub_dirs[current_sub_dir_index])))  # 当前子目录的实例数量
    current_instance_index = 0  # 当前实例索引

    while current_iteration < train_paras["max_iterations"]:
        # 每 x 次迭代更换实例
        if current_iteration % train_paras["parallel_iter"] == 0:
            case = []
            if current_instance_index >= num_instances - 1:
                current_sub_dir_index += 1  # 更换到下一个子目录
                if current_sub_dir_index >= len(sub_dirs):  
                    break
                current_instance_index = 0  # 重置实例索引
                config_file = os.path.join(os.path.join(root_dir, sub_dirs[current_sub_dir_index]), 'config_layout.json')  # 读取布局配置文件
            else:
                config_file = os.path.join(os.path.join(root_dir, sub_dirs[current_sub_dir_index]), 'config_layout.json')  # 读取布局配置文件

            while len(case) < env_paras["batch_size"]:
                if current_instance_index >= num_instances - 1:
                    current_sub_dir_index += 1  # 更换到下一个子目录
                    if current_sub_dir_index >= len(sub_dirs):
                        print("break")  
                        break
                    current_instance_index = 0  # 重置实例索引
                    config_file = os.path.join(os.path.join(root_dir, sub_dirs[current_sub_dir_index]), 'config_layout.json')  # 读取布局配置文件

                current_instance = os.path.join(root_dir, sub_dirs[current_sub_dir_index], 
                                            os.listdir(os.path.join(root_dir, sub_dirs[current_sub_dir_index]))[current_instance_index])
                
                if current_instance.endswith('.json') and current_instance != config_file:
                    case.append(current_instance)
                current_instance_index += 1  # 移动到下一个实例
            
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras, layout=config_file)
            current_instance_index += 1

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while not done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
        print("spend_time: ", time.time()-last_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error")
        print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy
        if current_iteration % train_paras["update_time_step"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([current_iteration]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([current_iteration]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  

        # if iter mod x = 0 then validate the policy
        if current_iteration % train_paras["save_time_step"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}.pt'.format(save_path, current_iteration)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([current_iteration]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))
        print("current_iteration: ", current_iteration + 1)
        current_iteration += 1  # 迭代次数更新
    print("训练完成")
    print("total_time: ", time.time() - start_time)


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    main()