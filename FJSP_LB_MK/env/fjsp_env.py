import sys
import gym
import torch
import time
import json
from dataclasses import dataclass
from env.load_data import graph_extract, nums_extract,all_plate_all_part_type,\
    every_plate_all_part_type,build_one_hot,compute_augmented_feat_opes,count_box_switches
from env.load_buffer_data import load_buffer,fenjian_ope,load_parts,Box

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_index, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_index
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, layout):
        '''
        :param case: The information of the plates
        :param env_paras: A dictionary of parameters for the environment
        '''
        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_machines = env_paras["num_machine"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        self.layout = read_json(layout)
        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        self.case=[]

        for i in range(self.batch_size):
            # print("case[i]=",case[i])
            line = read_json(case[i])
            lines.append(line)
            num_jobs, num_machine, num_opes = nums_extract(lines[i], self.layout)  ##for hengli, 8 ope
            self.num_opes = max(self.num_opes, num_opes)
            self.case.append(case[i])

        # load features
        for i in range(self.batch_size):
            load_data = graph_extract(lines[i], self.layout, num_machine, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])
        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
        # static feats
        # shape: (batch_size, num_opes, num_opes)
        # 前置工序的依赖关系，用于确定某个工序是否具备开始条件
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        # 后置工序的依赖关系，用于判断下一个工序何时可以启动
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        max_length1 = max(tensor.size(0) for tensor in tensors[5])  # 找到最长张量的长度
        padded_tensors1 = [F.pad(tensor, (0, max_length1 - tensor.size(0))) for tensor in tensors[5]]
        self.num_ope_biases_batch = torch.stack(padded_tensors1, dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        max_length2 = max(tensor.size(0) for tensor in tensors[6])  # 找到最长张量的长度
        padded_tensors2 = [F.pad(tensor, (0, max_length2 - tensor.size(0))) for tensor in tensors[6]]
        self.nums_ope_batch = torch.stack(padded_tensors2, dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_machine))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)  # 工序可选机器数量
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9) # 工序平均加工时间
        ###工序加工时间 for ope 6(fenjian) will change as the ... LB-MK problem
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch) # 工件尚未调度的工序数量
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),self.cal_cumul_adj_batch).squeeze() # 工序的开始时间
        self.end_ope_biases_batch = torch.clamp(self.end_ope_biases_batch, min=0)
        end_time_batch = (feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch) # 完工时间
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch) # 工序的完工时间
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1) # 该机器可以加工的工序数量
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        ### add in 0417
        self.scene_name = '恒力利材线'  ## 恒力利材线 大连重工 招商威海  芜湖船厂
        self.fenjian_ope, self.total_ope, self.fenjian_ma, \
        self.num_machines, self.max_part_type_per_batch = fenjian_ope(self.scene_name)
        all_part_type_file_path = '/home/zss/data/00_FJSP/25_0812_hgnn_real_data/' \
                                  'dataset/train_real_data/'+self.scene_name+\
                                  '_all_part_type/all_part_type.json'
        # 从文件中读取结果
        # with open(all_part_type_file_path, 'r') as infile:
        #     all_part_type = json.load(infile)
        # 获取 unique_part_attri_list
        # print("all_part_type=",all_part_type) ## 247 hengli
        # print("self.case=",self.case)
        all_plate_part_type=[[] for _ in range(self.batch_size)]
        self.every_plate_part_type = [[] for _ in range(self.batch_size)]
        for i_batch in range(self.batch_size):
            all_plate_part_type[i_batch]=all_plate_all_part_type(self.case[i_batch])
            self.every_plate_part_type[i_batch]=every_plate_all_part_type(self.case[i_batch])
            # print("every_plate_part_type[i_batch]=",every_plate_part_type[i_batch])
            # print("len(all_plate_part_type[i_batch])",len(all_plate_part_type[i_batch]))
        # print("all_plate_part_type=",all_plate_part_type) ## 28, 34
        # print("every_plate_part_type=",every_plate_part_type)

        ### 0902 build one-hot encodings
        # self.one_hot_encodings = build_one_hot(all_plate_part_type, self.every_plate_part_type, self.max_part_type_per_batch)
        # print("self.one_hot_encodings=",self.one_hot_encodings)
        # print("every_plate_part_type=",self.every_plate_part_type)
        # one_hot_encodings = torch.tensor(self.one_hot_encodings)
        # ope_onehots=[[] for _ in range(self.batch_size)]
        # for b in range(self.batch_size):
        #     nums_opes = self.nums_ope_batch[b]  # shape: [num_jobs]
        #     job_onehots = one_hot_encodings[b]
        #
        #     # print("nums_opes=",nums_opes)
        #     for j, num in enumerate(nums_opes.tolist()):
        #         repeated = job_onehots[j].unsqueeze(0).repeat(num, 1)  # [num, 10]
        #         ope_onehots[b].append(repeated)
        # ope_onehots_tensor = torch.stack([torch.stack(batch) for batch in ope_onehots])
        # print("ope_onehots_tensor.shape=",ope_onehots_tensor.shape)
        ### 0902 part-one-hot feat
        # self.feat_opes_batch = compute_augmented_feat_opes(self.feat_opes_batch, self.nums_ope_batch,
        #                                                    self.one_hot_encodings, self.max_part_type_per_batch)
        # print("self.feat_opes_batch=",self.feat_opes_batch)
        # print("self.feat_opes_batch[:, 7, :]=",self.feat_opes_batch[:, 7, :])
        ## """初始化工位上的料框"""
        ## hengli:6, dalian: how many gongwei total

        # print("self.fenjian_ope=",self.fenjian_ope) #{'bigger': 1000, 'large': 1000, 'medium': 1000, 'small': 6, 'tiny': 1000}
        # print("self.fenjian_ma=",self.fenjian_ma)   ##{'bigger': [], 'large': [], 'medium': [], 'small': [12, 13, 14], 'tiny': []}
        # num_fenjian = int(feat_opes_batch[:, 1, :][0][self.fenjian_ope])
        self.start_indices = torch.cumsum(
            torch.cat([
                torch.zeros(self.batch_size, 1, dtype=torch.long),
                self.nums_ope_batch[:, :-1]
            ], dim=1),
            dim=1
        )  # shape: [batch_size, num_jobs]
        # print("self.start_indices=",self.start_indices) #self.fenjian_ope= {'bigger': 6, 'large': 5, 'medium': 1000, 'small': 4, 'tiny': 1000}

        ### 0902 is_fenjian feat
        # offsets = torch.tensor([v for v in self.fenjian_ope.values() if v != 1000])  # [K]
        # # print("offsets=",offsets)
        # # 广播相加，得到 [B, J, K]
        # fenjian_ops = self.start_indices[..., None] + offsets[None, None, :]
        # self.all_fenjian_opes = fenjian_ops.reshape(self.start_indices.shape[0], -1)
        # B, J = self.all_fenjian_opes.shape
        # N = self.feat_opes_batch.shape[-1]
        # is_fenjian_mask = torch.zeros((B, N), device=self.all_fenjian_opes.device)
        # batch_idx = torch.arange(B).unsqueeze(1).expand(B, J)  # [B, J]
        # fenjian_idx = self.all_fenjian_opes  # [B, J]
        # is_fenjian_mask[batch_idx, fenjian_idx] = 1.0
        # self.feat_opes_batch[:, 6, :] = is_fenjian_mask
        # print("self.feat_opes_batch[:, 6, :]=",self.feat_opes_batch[:, 6, :])
        # print("self.all_fenjian_opes=",self.all_fenjian_opes)
        # print("job_type_feats=",job_type_feats)
        # 得到每个 job 的第3个工序的全局编号（起始索引 + 3）
        # self.third_opes = self.start_indices + self.fenjian_ope

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_machine), dtype=torch.bool, fill_value=False)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_machines, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_machines))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.total_switch_times={'bigger':0, 'large':0, 'medium': 0, 'small': 0,'tiny':0}
        self.batch_total_switch_times=[{'bigger':0, 'large':0, 'medium': 0, 'small': 0,'tiny':0}
                                       for _ in range(self.batch_size)]
        self.old_batch_total_switch_times = [{'bigger': 0, 'large': 0, 'medium': 0, 'small': 0, 'tiny': 0} for _ in
                                             range(self.batch_size)]


        self.total_part_type={'bigger':[], 'large': [], 'medium': [], 'small': [],'tiny':[]}
        self.max_total_part_type={'bigger':0, 'large': 0, 'medium': 0, 'small': 0,'tiny':0}
        self.min_total_part_type={'bigger':100, 'large': 100, 'medium':100, 'small': 100,'tiny':100}
        self.min_total_type=1000
        self.max_total_type=0

        self.total_part_num =[{'bigger':0, 'large':0, 'medium': 0, 'small': 0,'tiny':0}
                                       for _ in range(self.batch_size)]
        self.average_box_utiliz=0  ##
        # self.workstation={'bigger':[], 'large': [], 'medium': [], 'small': [],'tiny':[]}
        self.workstation=[[] for _ in range(self.batch_size)] ###only shared workstation
        # self.all_box={'large': [], 'medium': [], 'small': []}

        ### make a workstation for each fenjian machine
        for i_batch in range(self.batch_size):
            if self.scene_name != '恒力利材线':
                workstation, box_param = load_buffer(self.scene_name)
                for box_size in box_param[-1]:  ## add all type box size
                    for _ in range(box_param[-1][box_size]):
                        box = Box(0, box_size, box_param[1][box_size],
                                  box_param[2][box_size], box_param[3][box_size])
                        workstation.boxes[box_size].append(box)
                self.workstation[i_batch].append(workstation)
            else:
                for ma_id in range(len(self.fenjian_ma['small'])):
                    workstation, box_param = load_buffer(self.scene_name)
                    for box_size in box_param[-1]:  ## add all type box size
                        for _ in range(box_param[-1][box_size]):
                            box = Box(ma_id, box_size, box_param[1][box_size],
                                      box_param[2][box_size], box_param[3][box_size])
                            workstation.boxes[box_size].append(box)
                    self.workstation[i_batch].append(workstation)
        self.box_current_type_batch = [[[] for i in range(len(self.fenjian_ma['small']))]
                                       for _ in range(self.batch_size)]

        # print("self.workstation[0].boxes=",self.workstation[0].boxes) ## 2 large 4 medium 18 small
        # self.total_switch_times = {'bigger': 0, 'large': 0, 'medium': 0, 'small': 0, 'tiny': 0}
        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)
        self.old_workstation=copy.deepcopy(self.workstation)
        self.old_batch_total_switch_times=copy.deepcopy(self.batch_total_switch_times)
        self.action_list=[]
        self.instruction_list=[]

    def step(self, actions):
        # print("self.batch_size=",self.batch_size)
        '''
        Environment transition function
        '''
        self.action_list.append(actions)
        operations = actions[0, :]  ##should check if it's the fenjian operation (ope 6)
        # print("operations",operations)
        machines = actions[1, :]
        jobs = actions[2, :]
        # print("actions=",actions)
        # print("jobs=",jobs)
        # print("self.case=",self.case)
        self.N += 1
        # flag_fenjian = {'bigger':0, 'large': 0, 'medium': 0, 'small':0,'tiny':0}
        fenjian_ma = [0]*self.batch_size
        fenjian_size=[None]*self.batch_size
        self.delta_switch = [0] * self.batch_size
        # self.fenjian_ope= {'bigger': 6, 'large': 5, 'medium': 1000, 'small': 4, 'tiny': 1000}
        ##self.fenjian_ma= {'bigger': [11], 'large': [10], 'medium': [], 'small': [9], 'tiny': []}
        for i_batch in range(self.batch_size):
            for size in self.fenjian_ope:
                if self.fenjian_ope[size] != 1000 and \
                        (int(operations[i_batch]) - self.fenjian_ope[size]) % self.total_ope == 0:
                    ##the ope belong to 超大件分拣, 大件分拣, ...
                    # flag_fenjian[size] = 1
                    fenjian_size[i_batch] = size
                    # print("fenjian_size[i_batch]=",fenjian_size[i_batch])
                    for ma in range(len(self.fenjian_ma[size])):
                        if int(machines[i_batch]) == self.fenjian_ma[size][ma]:
                            # print("machines[i_batch]=",machines[i_batch])
                            fenjian_ma[i_batch] = ma
                            break

        # Ensure all tensors are on the same device
        self.ope_ma_adj_batch = self.ope_ma_adj_batch.to(self.device)
        self.proc_times_batch = self.proc_times_batch.to(self.device)
        self.batch_idxes = self.batch_idxes.to(self.device)
        self.feat_opes_batch = self.feat_opes_batch.to(self.device)
        self.num_ope_biases_batch = self.num_ope_biases_batch.to(self.device)
        self.end_ope_biases_batch = self.end_ope_biases_batch.to(self.device)
        self.time = self.time.to(self.device)
        self.cal_cumul_adj_batch = self.cal_cumul_adj_batch.to(self.device)
        self.opes_appertain_batch = self.opes_appertain_batch.to(self.device)
        self.schedules_batch = self.schedules_batch.to(self.device)
        self.machines_batch = self.machines_batch.to(self.device)
        self.feat_mas_batch = self.feat_mas_batch.to(self.device)

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_machines), dtype=torch.int64).to(self.device)
        remain_ope_ma_adj[self.batch_idxes, machines] = 1
        # print("Shape of remain_ope_ma_adj[self.batch_idxes, :]:", remain_ope_ma_adj[self.batch_idxes, :].shape)
        # print("Shape of self.ope_ma_adj_batch[self.batch_idxes, operations]:", self.ope_ma_adj_batch[self.batch_idxes, operations].shape)
        self.ope_ma_adj_batch[self.batch_idxes, operations] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch
        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, operations, machines].to(
            self.device)
        for i_batch in range(self.batch_size):
            if fenjian_size[i_batch] == None:  ## not fenjian
                # proc_times[i_batch] = self.proc_times_batch[i_batch, operations[i_batch], machines[i_batch]].to(
                #     self.device)  ##estimate to true
                pass
                # print("not fenjian")
            else:
                # print("is fenjian")
                parts = load_parts(self.case[i_batch], int(jobs[i_batch]))  ## one part by one part
                fixed_coef = 180
                parts_num = len(parts)
                # print("jobs[i_batch]=",jobs[i_batch])
                # print("parts_num=",parts_num)
                # print("self.parts_num=",self.parts_num)
                process_time = fixed_coef
                current_jobs_part_type = {'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny': []}
                current_jobs_total_type = []
                for i in range(parts_num):
                    # print("fenjian_size[{}]=".format(i_batch),fenjian_size[i_batch])
                    #fenjian_size[3]= large
                    if parts[i]['part_weight'] > 1000:  ###
                        continue
                    if parts[i]['part_attri'] not in current_jobs_total_type:
                        current_jobs_total_type.append(parts[i]['part_attri'])
                    if parts[i]['part_attri'] not in current_jobs_part_type[parts[i]['part_type']]:
                        current_jobs_part_type[parts[i]['part_type']].append(parts[i]['part_attri'])
                    if fenjian_size[i_batch] == 'bigger':  ##chaodajian can only set in large box
                        if parts[i]['part_type'] != 'bigger':
                            continue  ## only do 'bigger'part
                        self.total_part_num[i_batch][parts[i]['part_type']] += parts[i]['part_num']
                        if parts[i]['part_attri'] not in self.total_part_type[parts[i]['part_type']]:
                            self.total_part_type[parts[i]['part_type']].append(parts[i]['part_attri'])
                        if self.scene_name != '招商威海':
                            parts[i]['part_type'] = 'large'  ## change part_type to large (so that it can be added)
                    elif fenjian_size[i_batch] == 'large':  ## only do large and medium ope
                        if parts[i]['part_type'] != 'large' and parts[i]['part_type'] != 'medium':
                            continue
                        self.total_part_num[i_batch][parts[i]['part_type']] += parts[i]['part_num']
                        if parts[i]['part_attri'] not in self.total_part_type[parts[i]['part_type']]:
                            self.total_part_type[parts[i]['part_type']].append(parts[i]['part_attri'])
                        if self.scene_name != '大连重工':  ##'招商威海' and '芜湖船厂' has no medium size box
                            parts[i]['part_type'] = 'large'  ### adjust the 'medium' part type to 'large'
                    elif fenjian_size[i_batch] == 'small':
                        if parts[i]['part_type'] != 'small':  ## only do small
                            continue
                        self.total_part_num[i_batch][parts[i]['part_type']] += parts[i]['part_num']
                        if parts[i]['part_attri'] not in self.total_part_type[parts[i]['part_type']]:
                            self.total_part_type[parts[i]['part_type']].append(parts[i]['part_attri'])
                    else:
                        # print("else, fenjian_size=", fenjian_size)
                        if parts[i]['part_type'] != 'tiny':  ## only do tiny
                            continue
                        self.total_part_num[i_batch][parts[i]['part_type']] += parts[i]['part_num']
                        if parts[i]['part_attri'] not in self.total_part_type[parts[i]['part_type']]:
                            self.total_part_type[parts[i]['part_type']].append(parts[i]['part_attri'])
                        parts[i]['part_type'] = 'small'  ### adjust the 'tiny' part type to 'small'

                    current_time = copy.deepcopy(self.time[i_batch])
                    # print("parts[i]['part_type']=",parts[i]['part_type']) ##large
                    # print("fenjian_ma[i_batch]=",fenjian_ma[i_batch])
                    time_after_add_part = self.workstation[i_batch][fenjian_ma[i_batch]].ws_add_parts(size=parts[i]['part_type'],
                                                                                    num_parts=parts[i]['part_num'],
                                                                                    part_weight=parts[i]['part_weight'],
                                                                                    part_type=parts[i]['part_attri'],
                                                                                    time=current_time)
                    time_used = time_after_add_part - self.time[i_batch]
                    # print("time_used=",time_used)
                    process_time += time_used
                ##add in 0715
                self.batch_total_switch_times[i_batch] = {'bigger': 0, 'large': 0, 'medium': 0, 'small': 0, 'tiny': 0}
                # for fenjian_size in
                for ma in range(len(self.fenjian_ma[fenjian_size[i_batch]])):
                    for size in self.workstation[i_batch][ma].boxes:
                        #     size='small'
                        for i in range(len(self.workstation[i_batch][ma].boxes[size])):
                            # print("self.workstation[{}][{}].boxes[{}][{}].switch_times=".
                            #       format(i_batch,ma,size,i),
                            #       self.workstation[i_batch][ma].boxes[size][i].switch_times)
                            self.batch_total_switch_times[i_batch][size] += \
                                self.workstation[i_batch][ma].boxes[size][i].switch_times
                            # print("switch times for workstation {} size {} box {} is".format(ma,size, i),
                            #       self.workstation[ma].boxes[size][i].switch_times)
                self.delta_switch[i_batch] = 0
                for size in self.batch_total_switch_times[i_batch]:
                    current_size_switch= self.batch_total_switch_times[i_batch][size] \
                    - self.old_batch_total_switch_times[i_batch][size]
                    self.delta_switch[i_batch]+=current_size_switch
                self.old_batch_total_switch_times[i_batch] = self.batch_total_switch_times[i_batch]
                ### 0831 update the box current part type
                # self.box_current_type_batch[i_batch] = [[] for i in range(len(self.fenjian_ma['small']))]
                # for ma in range(len(self.fenjian_ma['small'])):
                #     for size in self.workstation[i_batch][ma].boxes:
                #         for i_box in range(len(self.workstation[i_batch][ma].boxes[size])):
                #             box_part_type_id = self.workstation[i_batch][ma].boxes[size][
                #                 i_box].current_part_type
                #             self.box_current_type_batch[i_batch][ma].append(box_part_type_id)
                # print("self.box_current_type_batch=",self.box_current_type_batch)
                if process_time == 0:
                    process_time = 1
                # print("process_time=", process_time)
                # proc_times = torch.tensor([1])
                proc_times[i_batch]= torch.tensor([process_time])

        # print("actions[0]=",actions[0])
        # print("actions[0][0]=",actions[0][0]) ## operation_in_total_num
        # print("actions[1][0]=", actions[1][0]) ## machine
        # print("actions[2][0]=", actions[2][0])  ## job
        current_instruction = [
            np.array(self.time.cpu()),
            np.array((actions[1][0] + 1).cpu()),
            np.array((actions[2][0] + 1).cpu()),
            np.array(proc_times.cpu())
        ]
        # print()
        # print("self.time=",self.time)
        print("current_instruction=",current_instruction)
        self.instruction_list.append(current_instruction)



        self.feat_opes_batch[self.batch_idxes, :3, operations] = torch.stack((
            torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
            torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
            proc_times
            ), dim=1)
        ### compute switch_times
        # print("self.box_current_type_batch=",self.box_current_type_batch)

        ### 0902 estimate box_switches
        # switches=count_box_switches(self.every_plate_part_type,self.box_current_type_batch)
        # # print("switches=",switches)
        # switches=torch.tensor(switches)
        # # print("switches=", switches.shape)  #torch.Size([2, 20, 3])
        # B, J, W = switches.shape
        # _, N, O = self.feat_opes_batch.shape
        #
        # expanded_switches = []
        # for b in range(B):
        #     job_switches = []
        #     for j in range(J):
        #         sw = switches[b, j]  # [W]
        #         job_switches.append(sw.unsqueeze(1).repeat(1, self.nums_ope_batch[b, j]))  # [W, num_ope_j]
        #     expanded_switches.append(torch.cat(job_switches, dim=1))  # [W, O_b]
        #
        # expanded_switches = torch.stack(expanded_switches, dim=0)  # [B, W, O]
        #
        # # 替换 feat_opes_batch 的最后 3 个特征维
        # self.feat_opes_batch[:, -W:, :] = expanded_switches
        # print("self.feat_opes_batch[:, -3:, :]=",self.feat_opes_batch[:, -3:, :])
        ##self.feat_opes_batch[:,2,:] will change from estimate value to true value(from juzhan code)
        last_opes = torch.where(operations - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs],
                                self.num_opes - 1, operations - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, operations] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]  ## mean to current
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                            self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes,:])
        # print("self.feat_opes_batch[self.batch_idxes, 4, :]=",self.feat_opes_batch[self.batch_idxes, 4, :]) ## end_time
        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, operations, :2] = torch.stack((torch.ones(self.batch_idxes.size(0), device=self.device), machines), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, machines, 0] = torch.zeros(self.batch_idxes.size(0), device=self.device)
        self.machines_batch[self.batch_idxes, machines, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, machines, 2] += proc_times
        self.machines_batch[self.batch_idxes, machines, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :].to(self.device), dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, machines] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        self.ope_step_batch = self.ope_step_batch.to(self.device)
        self.mask_job_finish_batch = self.mask_job_finish_batch.to(self.device)
        self.makespan_batch = self.makespan_batch.to(self.device)

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, machines] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max = torch.max(self.feat_opes_batch[:, 4, :].to(self.device), dim=1)[0]
        self.reward_batch = 1*(self.makespan_batch - max)
        self.makespan_batch = max
        for batch_id in range(self.batch_size):
            if fenjian_size[batch_id] != None:
                self.reward_batch[batch_id] += -10 * self.delta_switch[batch_id]  ##dense reward to

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        # print("flag_trans_2_next_time=",flag_trans_2_next_time)
        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()
        # print("flag_trans_2_next_time, after next time=",flag_trans_2_next_time)
        # Update the vector for uncompleted instances
        mask_finish = (self.N+1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch.to(self.device) + self.mask_job_finish_batch.to(self.device))[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible.to(self.device) & job_eligible.to(self.device), op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)
        self.workstation=copy.deepcopy(self.old_workstation)
        self.batch_total_switch_times=copy.deepcopy(self.old_batch_total_switch_times)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_machines), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_machines, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_machines))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    def render(self):
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_machines
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i + 1)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title(u'FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                             0.2,
                             left=schedules[id_ope][2],
                             color='#b2b2b2',
                             height=0.5)
                    axes.barh(id_machine,
                             schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                             left=schedules[id_ope][2]+0.2,
                             color=color[idx_job],
                             height=0.5)
                plt.show()
        return

    def render_new(self):
        '''
        Deprecated in the final experiment
        '''
        # print("render_new")
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_machines
            print(sys.argv[0])
            color = read_json("/home/zss/data/00_FJSP/25_0218_productionscheduling/utils/color_config.json")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "/home/zss/data/00_FJSP/25_0218_productionscheduling/utils/color_config.json")
            labels = [''] * num_jobs
            for j in range(num_jobs):
                labels[j] = "job {0}".format(j)
            # print("self.batch_size=",self.batch_size)
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                # print("schedules=", schedules)
                fig, ax = plt.subplots()
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(num_jobs)]
                ax.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=14)
                fig = ax.figure
                fig.set_size_inches(16, 8)
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(num_mas - 1 - i))
                    y_ticks_loc.insert(0, i)
                ax.set_yticks(y_ticks_loc, y_ticks, fontsize=16)
                # for i in range(num_mas):
                #     y_ticks.append('Machine {0}'.format(i))
                #     y_ticks_loc.insert(0, i)

                ax.set_xlabel('Time')
                ax.set_ylabel('Machine')
                ax.set_title('Job Shop Scheduling Gantt Chart')
                ax.grid(True)
                for i in range(num_mas - 1, -1, -1):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j)
                # patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                # axes.cla()
                # axes.set_title(u'FJSP Schedule',fontsize=18)
                # axes.grid(linestyle='-.', color='gray', alpha=0.2)
                # axes.set_xlabel('Time',fontsize=16)
                # axes.set_ylabel('Machine',fontsize=16)
                # axes.set_yticks(y_ticks_loc, y_ticks,fontsize=16)
                # axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                # axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                #
                # axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=16)
                # print("self.num_ope_biases_batch=", self.num_ope_biases_batch)
                # print("int(self.nums_opes[batch_id])=",int(self.nums_opes[batch_id]))
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    # print("id_ope=", id_ope)
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    # print("idx_job, idx_ope=", idx_job, idx_ope)
                    id_machine = schedules[id_ope][1]
                    # print("id_machine", id_machine)
                    id_machine = int(id_machine)
                    operation_start = int(schedules[id_ope][2])
                    operation_duration = int(schedules[id_ope][3]) - int(schedules[id_ope][2])

                    # axes.barh(id_machine,
                    #          0.2,
                    #          left=schedules[id_ope][2],
                    #          color='#b2b2b2',
                    #          height=0.5)
                    ax.broken_barh(
                        [(operation_start, operation_duration)],
                        (id_machine - 0.4, 0.8),
                        facecolors=color[idx_job],
                        edgecolor='black'
                    )
                    # ax.barh(id_machine,
                    #          schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                    #          left=schedules[id_ope][2]+0.2,
                    #          color=color[idx_job],
                    #          height=0.5)
                    # ax.text(0.5*(schedules[id_ope][2]+schedules[id_ope][3]),schedules[id_ope][1],f'{id_ope}',ha='center',va='bottom',fontsize=14)
                    middle_of_operation = operation_start + operation_duration / 2
                    ax.text(
                        middle_of_operation,
                        id_machine,
                        id_ope,
                        ha='center',
                        va='center',
                        fontsize=16
                    )
                plt.savefig("./gantt_result_0416/test_result_{0}".format(str_time))
                # plt.show()

        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_machines)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_machines):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2] > ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if not torch.isclose(torch.tensor(ma_gantt[i][j][2] - ma_gantt[i][j][1]), proc_time[ma_gantt[i][j][0]][i], atol=1e-3):
                        # print(torch.tensor(ma_gantt[i][j][2] - ma_gantt[i][j][1]))
                        # print(proc_time[ma_gantt[i][j][0]][i])
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass


