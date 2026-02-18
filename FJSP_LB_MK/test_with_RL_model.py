import copy
import json
import os
import random
import time as time
import re
import gym
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model
import PPO_model_KL_reg
from env.load_data import nums_extract
from utils.my_utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config_train_file.json", 'r') as load_f:
        load_dict = json.load(load_f)
        layout = {}
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    mod_files = os.listdir('./model/')[:]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    # 生成数据文件并写入表头
    name = "恒力利材线"
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save_all_dataset/{1}_{0}'.format(str_time,name)
    os.makedirs(save_path)
    # file_path = "dataset/恒力利材线_20/"
    # file_path = "dataset/大连重工_20/"  #招商威海 #芜湖船厂
    file_path="dataset/DataWarehouse-master/数据集/plate_part/"+name+"/"
    test_instances = []
    test_file = os.listdir(file_path)
    for i_ins in range(len(test_file)):
        if test_file[i_ins].endswith('.json') and test_file[i_ins] != "config_layout.json":
            test_instances.append(test_file[i_ins])
    test_instances = sorted(os.listdir(file_path),
                            key=lambda x: [int(num) for num in re.findall(r'\d+', x)])
    # num_ins = len(test_instances)
    num_ins=1

    # 创建 makespan 文件并写入表头
    file_name = [test_instances[i] for i in range(num_ins)]
    # file_name = [test_instances[i] for i in range(3,4)]

    data_file = pd.DataFrame(file_name, columns=["file_name"])
    with pd.ExcelWriter('{0}/makespan_{1}.xlsx'.format(save_path, str_time)) as writer:
        data_file.to_excel(writer, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time)) as writer_time:
        data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter('{0}/part_type_{1}.xlsx'.format(save_path, str_time)) as writer_ave_part_type:
        data_file.to_excel(writer_ave_part_type, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter('{0}/max_part_type_{1}.xlsx'.format(save_path, str_time)) as writer_ave_max_part_type:
        data_file.to_excel(writer_ave_max_part_type, sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('{0}/min_part_type_{1}.xlsx'.format(save_path, str_time)) as writer_ave_min_part_type:
        data_file.to_excel(writer_ave_min_part_type, sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('{0}/max_total_type_{1}.xlsx'.format(save_path, str_time)) as writer_ave_max_total_type:
        data_file.to_excel(writer_ave_max_total_type, sheet_name='Sheet1', index=False)
    with pd.ExcelWriter('{0}/min_total_type_{1}.xlsx'.format(save_path, str_time)) as writer_ave_min_total_type:
        data_file.to_excel(writer_ave_min_total_type, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter('{0}/part_num_{1}.xlsx'.format(save_path, str_time)) as writer_ave_part_num:
        data_file.to_excel(writer_ave_part_num, sheet_name='Sheet1', index=False)

    with pd.ExcelWriter('{0}/switch_times_{1}.xlsx'.format(save_path, str_time)) as writer_ave_switch_times:
        data_file.to_excel(writer_ave_switch_times, sheet_name='Sheet1', index=False)

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    startcol = 1
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                # model_CKPT = torch.load('./model/' + mod_files[i_rules]) ## original
                ## Hengli
                # model_CKPT = torch.load('/home/zss/data/00_FJSP/25_0812_hgnn_real_data/save_train_file/train_20250904_184007/save_best_9070.pt')
                ## Weihai
                # model_CKPT =torch.load('/home/zss/data/00_FJSP/25_0812_hgnn_real_data/save_train_file/train_20250906_234652/save_best_2080.pt')
                ## Dalian
                # model_CKPT=torch.load('/home/zss/data/00_FJSP/25_0812_hgnn_real_data/save_train_file/train_20250909_234820/save_best_50.pt')
                ## Wuhu
                model_CKPT=torch.load('/home/zss/data/00_FJSP/25_0812_hgnn_real_data/save_train_file/train_20250908_191346/save_best_3090.pt')
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        switch_small = []
        times = []
        all_part_type={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}  ##small,medium,large
        all_switch_times={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
        all_part_num={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
        all_max_part_type = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
        all_min_part_type = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
        all_max_total_type = []
        all_min_total_type = []
        # config_file = 'dataset/恒力利材线_layout.json'
        config_file = 'dataset/DataWarehouse-master/数据集/plate_part/'+name+'_layout.json'
        layout = read_json(config_file)
        min_process_time = 10000
        max_process_time = 0
        for i_ins in range(8,9): ##num_ins=1
            if test_instances[i_ins].endswith('.json') and test_instances[i_ins] != "config_layout.json":
                instance = file_path + test_instances[i_ins]
                ins_num_jobs, ins_num_mas, _ = nums_extract(read_json(instance), layout)
                env_test_paras["num_jobs"] = ins_num_jobs
                env_test_paras["num_mas"] = ins_num_mas
            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = gym.make('fjsp-v0', case=[instance] * test_paras["num_sample"],
                                   env_paras=env_test_paras, layout=config_file)
                # DRL-G, each env contains one instance
                else:
                    print("instance=",instance)
                    env = gym.make('fjsp-v0', case=[instance], 
                                   env_paras=env_test_paras, layout=config_file)
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                time_s = []
                makespan_s = []
                part_type_s = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}  ## small,medium,large
                part_num_s = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
                max_part_type_s = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
                min_part_type_s = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
                max_total_type_s = []
                min_total_type_s = []
                switch_times_s = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
                for j in range(test_paras["num_average"]):
                    start_schedule = time.time()
                    makespan, time_re, part_type, part_num, switch_times, max_part_type, \
                    min_part_type, max_total_type, min_total_type,process_time,df = schedule(env, model, memories, flag_sample=test_paras["sample"])
                    # print("schedule_dur=",time.time()-start_schedule)
                    print("part_type=", part_type)
                    print("switch_times=", switch_times)
                    df.to_excel(save_path+'/instruction_data.xlsx', index=False)

                    for size in part_type:
                        part_type_s[size].append(float(len(part_type[size])))
                        part_num_s[size].append(float(part_num[0][size]))
                        switch_times_s[size].append(float(switch_times[0][size]))
                        max_part_type_s[size].append(float(max_part_type[size]))
                        min_part_type_s[size].append(float(min_part_type[size]))
                    max_total_type_s.append(float(max_total_type))
                    min_total_type_s.append(float(min_total_type))
                    print("part_num=", part_num)
                    print("switch_times=", switch_times)
                    sums = [sum(d.values()) for d in switch_times]
                    # print("sums=",sums)
                    makespans.append(torch.min(makespan))
                    min_idx = torch.argmin(makespan)
                    switch_small.append(sums[min_idx])
                    print("switch_small=",switch_small)

                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    # part_type_s.append(part_type)
                    # switch_times_s.append(switch_times)
                    env.reset()
                # makespans.append(torch.mean(torch.tensor(makespan_s)))
                for size in all_part_type:
                    all_part_type[size].append(torch.mean(torch.tensor(part_type_s[size])))
                    all_part_num[size].append(torch.mean(torch.tensor(part_num_s[size])))
                    all_switch_times[size].append(torch.mean(torch.tensor(switch_times_s[size])))
                    all_max_part_type[size].append(torch.mean(torch.tensor(max_part_type_s[size])))
                    all_min_part_type[size].append(torch.mean(torch.tensor(min_part_type_s[size])))
                all_max_total_type.append(torch.mean(torch.tensor(max_total_type_s)))
                all_min_total_type.append(torch.mean(torch.tensor(min_total_type_s)))
                # ave_part_type.append(torch.mean(torch.tensor(part_type_s)))
                # ave_switch_times.append(torch.mean(torch.tensor(switch_times_s)))
                print("rule for {}, makespans={}".format(rule, makespans))
                average_makespan = torch.mean(torch.tensor(makespans))
                ave_switch = np.mean(switch_small)
                print("ave_switch=", ave_switch)
                # average_part_type = torch.mean(torch.tensor(ave_part_type))
                # average_switch_times = torch.mean(torch.tensor(ave_switch_times))
                print("average_makespan=", average_makespan)
                # print("average_part_type=", average_part_type)
                # print("average_switch_times=", average_switch_times)
                times.append(torch.mean(torch.tensor(time_s)))

            # DRL-G
            else:
                time_s = []
                makespan_s = []
                part_type_s= {'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []} ## small,medium,large
                part_num_s={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
                max_part_type_s={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
                min_part_type_s={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
                max_total_type_s = []
                min_total_type_s = []

                switch_times_s={'bigger': [],'large': [], 'medium': [], 'small': [],'tiny': []}
                for j in range(test_paras["num_average"]):
                    start_schedule=time.time()
                    makespan,time_re,part_type,part_num,switch_times,max_part_type,\
                min_part_type,max_total_type,min_total_type,process_time,df= schedule(env, model, memories)
                    # print("schedule_dur=",time.time()-start_schedule)
                    df.to_excel(save_path+'/instruction_data.xlsx', index=False)
                    print("part_type=",part_type)
                    print("switch_times=",switch_times)
                    mask = process_time != 0
                    current_min_process_time = process_time[mask].min()
                    current_max_process_time = process_time.max()
                    if current_max_process_time > max_process_time:
                        max_process_time=current_max_process_time
                    if current_min_process_time < min_process_time:
                        min_process_time=current_min_process_time
                    for size in part_type:
                        part_type_s[size].append(float(len(part_type[size])))
                        part_num_s[size].append(float(part_num[0][size]))
                        switch_times_s[size].append(float(switch_times[0][size]))
                        max_part_type_s[size].append(float(max_part_type[size]))
                        min_part_type_s[size].append(float(min_part_type[size]))
                    max_total_type_s.append(float(max_total_type))
                    min_total_type_s.append(float(min_total_type))
                    print("part_num=",part_num)
                    print("switch_times=",switch_times)
                    # part_type=float(part_type)
                    # switch_times=float(switch_times)

                    # print("part_type=",part_type)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    # part_type_s.append(part_type)
                    # switch_times_s.append(switch_times)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                for size in all_part_type:
                    all_part_type[size].append(torch.mean(torch.tensor(part_type_s[size])))
                    all_part_num[size].append(torch.mean(torch.tensor(part_num_s[size])))
                    all_switch_times[size].append(torch.mean(torch.tensor(switch_times_s[size])))
                    all_max_part_type[size].append(torch.mean(torch.tensor(max_part_type_s[size])))
                    all_min_part_type[size].append(torch.mean(torch.tensor(min_part_type_s[size])))
                all_max_total_type.append(torch.mean(torch.tensor(max_total_type_s)))
                all_min_total_type.append(torch.mean(torch.tensor(min_total_type_s)))
                # ave_part_type.append(torch.mean(torch.tensor(part_type_s)))
                # ave_switch_times.append(torch.mean(torch.tensor(switch_times_s)))
                print("rule for {}, makespans={}".format(rule,makespans))
                average_makespan=torch.mean(torch.tensor(makespans))
                # average_part_type = torch.mean(torch.tensor(ave_part_type))
                # average_switch_times = torch.mean(torch.tensor(ave_switch_times))
                print("average_makespan=",average_makespan)
                # print("average_part_type=", average_part_type)
                # print("average_switch_times=", average_switch_times)
                times.append(torch.mean(torch.tensor(time_s)))

            print("finish env {0}".format(i_ins))
        print("rule_spend_time: ", time.time() - step_time_last)
        print("min_process_time=", min_process_time)
        print("max_process_time=", max_process_time)
        # 保存 makespan 和 time 数据到文件
        with pd.ExcelWriter('{0}/makespan_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer:
            data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
            data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=startcol)
            startcol+=1
        with pd.ExcelWriter('{0}/min_total_type_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_ave_min_total_type:
            data = pd.DataFrame(torch.tensor(all_min_total_type).t().tolist(), columns=[rule])
            data.to_excel(writer_ave_min_total_type, sheet_name='Sheet1', index=False, startcol=startcol)
            startcol+=1
        with pd.ExcelWriter('{0}/max_total_type_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_ave_max_total_type:
            data = pd.DataFrame(torch.tensor(all_max_total_type).t().tolist(), columns=[rule])
            data.to_excel(writer_ave_max_total_type, sheet_name='Sheet1', index=False, startcol=startcol)
            startcol+=1

        with pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_time:
            data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
            data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=startcol)
            startcol += 1

        with pd.ExcelWriter('{0}/part_type_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_ave_part_type:
            for i_size in all_part_type:
                data_part_type=pd.DataFrame(torch.tensor(all_part_type[i_size]).t().tolist(), columns=[i_size])
                data_part_type.to_excel(writer_ave_part_type, sheet_name='Sheet1', index=False, startcol=startcol)
                startcol += 1
        with pd.ExcelWriter('{0}/max_part_type_{1}.xlsx'.format(save_path, str_time), mode='a',
                                engine='openpyxl', if_sheet_exists='overlay') as writer_ave_max_part_type:
            for i_size in all_max_part_type:
                data_part_type = pd.DataFrame(torch.tensor(all_max_part_type[i_size]).t().tolist(), columns=[i_size])
                data_part_type.to_excel(writer_ave_max_part_type, sheet_name='Sheet1', index=False, startcol=startcol)
                startcol += 1
        with pd.ExcelWriter('{0}/min_part_type_{1}.xlsx'.format(save_path, str_time), mode='a',
                                engine='openpyxl', if_sheet_exists='overlay') as writer_ave_min_part_type:
            for i_size in all_min_part_type:
                data_part_type = pd.DataFrame(torch.tensor(all_min_part_type[i_size]).t().tolist(),
                                                  columns=[i_size])
                data_part_type.to_excel(writer_ave_min_part_type, sheet_name='Sheet1', index=False,
                                            startcol=startcol)
                startcol += 1
            # data = pd.DataFrame(torch.tensor(ave_part_type).t().tolist(), columns=[rule])

        with pd.ExcelWriter('{0}/part_num_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_ave_part_num:
            for i_size in all_part_num:
                data_part_num = pd.DataFrame(torch.tensor(all_part_num[i_size]).t().tolist(), columns=[i_size])
                data_part_num.to_excel(writer_ave_part_num, sheet_name='Sheet1', index=False, startcol=startcol)
                startcol += 1

        with pd.ExcelWriter('{0}/switch_times_{1}.xlsx'.format(save_path, str_time),mode='a',
                            engine='openpyxl', if_sheet_exists='overlay') as writer_ave_switch_times:
            for i_size in all_switch_times:
                data_switch_times = pd.DataFrame(torch.tensor(all_switch_times[i_size]).t().tolist(), columns=[i_size])
                data_switch_times.to_excel(writer_ave_switch_times, sheet_name='Sheet1', index=False, startcol=startcol)
                startcol += 1

        for env in envs:
            env.reset()

    print("total_spend_time: ", time.time() - start)

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    while not done:
        start_action = time.time()
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        # print("model predict/act time=", time.time() - start_action)
        start_step=time.time()
        state, rewards, dones = env.step(actions)  # environment transit
        # print("step time=",time.time()-start_step)
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    print("spend_time: ", spend_time)
    # print("env.action_list=",env.action_list)
    # print("env.instruction_list=",env.instruction_list)
    instruct_data = []
    for i, instruction in enumerate(env.instruction_list):
        row = {
            'Step': i + 1,
            'Time': instruction[0][0] if isinstance(instruction[0], np.ndarray) else instruction[0],
            'Machine_ID': instruction[1] if isinstance(instruction[1], np.ndarray) else instruction[1],
            'Job_ID': instruction[2] if isinstance(instruction[2], np.ndarray) else instruction[2],
            'Process_Time': instruction[3][0] if isinstance(instruction[3], np.ndarray) else instruction[3]
        }
        instruct_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(instruct_data)

    # Save to Excel file
    # df.to_excel('instruction_data.xlsx', index=False)

    # print("Data has been saved to 'instruction_data.xlsx'")

    # name = "大连重工"
    # str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    # save_path = './save_all_dataset/{1}_{0}'.format(str_time, name)
    # os.makedirs(save_path)
    # # torch.save(env.action_list,save_path+"/action_list.pt")
    render_start=time.time()
    # env.render_new()

    # print("render_spend=",time.time()-render_start)
    # Verify the solution
    # gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error")
    return copy.deepcopy(env.makespan_batch), spend_time,\
           copy.deepcopy(env.total_part_type),copy.deepcopy(env.total_part_num),\
           copy.deepcopy(env.batch_total_switch_times),copy.deepcopy(env.max_total_part_type),\
           copy.deepcopy(env.min_total_part_type),copy.deepcopy(env.max_total_type),\
           copy.deepcopy(env.min_total_type),copy.deepcopy(env.proc_times_batch),df



if __name__ == '__main__':
    main()