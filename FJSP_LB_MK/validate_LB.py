import gym
import env
import PPO_model_train_file
import PPO_model
import PPO_model_KL_reg
import torch
import time
import os
import copy
from utils.my_utils import *
import re
scene_name="大连重工"  ### 恒力利材线 大连重工 招商威海  芜湖船厂
from env.load_data import nums_extract
def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    # file_path ="./dataset/DataWarehouse-master/数据集/plate_part/"+scene_name+"/"
    file_path="dataset/DataWarehouse-master/数据集/plate_part/"+scene_name+"/"
    config_file = '/home/zss/data/00_FJSP/25_0502_FJSP_LB/dataset' \
                  '/DataWarehouse-master/数据集/plate_part/' + scene_name + '_layout.json'
    layout = read_json(config_file)
    # valid_data_files = os.listdir(file_path)
    # plate_parts_files = os.listdir(plate_part_path)
    # plate_parts_type_files = os.listdir(plate_part_type_path)
    valid_data_files=sorted(os.listdir(file_path),
                            key=lambda x: [int(num) for num in re.findall(r'\d+', x)])
    for i in range(len(valid_data_files)): #len(valid_data_files)  20
        valid_data_files[i] = file_path+valid_data_files[i]
    ins_num_jobs, ins_num_mas, _ = nums_extract(read_json(valid_data_files[0]), layout)
    env_paras["num_jobs"] = ins_num_jobs
    env_paras["num_mas"] = ins_num_mas

    # ins_num_jobs, ins_num_mas, _ = nums_extract(read_json(valid_data_files[0]), layout)
    # env_paras["num_jobs"] = ins_num_jobs
    # env_paras["num_mas"] = ins_num_mas
    # print("valid_data_files=",valid_data_files)
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras,layout=config_file)
    return env

def validate(env_paras, env, model_policy):
    switch_time = {'bigger': 0, 'large': 0, 'medium': 0, 'small': 0, 'tiny': 0}
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model_KL_reg.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    # gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    all_switch_time=[]

    for batch_id in range(batch_size):
        total_switch_time = 0
        for size in switch_time:
            # if size=="small":
            # print("size=",size)
            all_current_size_switch_time=copy.deepcopy(env.batch_total_switch_times[batch_id][size])
            # print("all_current_size_switch_time=",all_current_size_switch_time)
            total_switch_time+=all_current_size_switch_time
                # print("all_switch_time=", all_switch_time)
        all_switch_time.append(total_switch_time)
                # print("all_switch_time=",all_switch_time)
            # aa=env.total_switch_times_batch[batch_id][size]
            # print("batch {} size {}, total_switch_time=".format(batch_id,size),aa)

    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    print(" len(all_switch_time)=", len(all_switch_time)) ##100
    print(" sum(all_switch_time)=", sum(all_switch_time)) ##174
    print("all_switch_time=",all_switch_time)
    ave_switch_times=copy.deepcopy(sum(all_switch_time) / len(all_switch_time))

    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch,ave_switch_times
