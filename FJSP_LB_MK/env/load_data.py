import torch
import numpy as np
import math
import json
class Box:
    def __init__(self, size, length, width, max_capacity,type=None):
        self.size = size  # 料框尺寸：'large', 'medium', 'small'
        self.length = length  # 长度
        self.width = width  # 宽度
        self.max_capacity = max_capacity  # 最大容量
        self.current_capacity = 0  # 当前存放的零件总质量
        self.current_part_type=type   ## 当前存放的零件Type (同类零件才可以放到一个框里, init as None)

    def add_parts(self, num_parts, part_type):
        """向料框中添加零件"""
        if self.current_capacity + num_parts <= self.max_capacity and \
                (part_type == self.current_part_type or self.current_part_type == None):
            self.current_capacity += num_parts
            self.current_part_type = part_type  # 原来是个空框，None变part_type
            return num_parts  # 返回成功添加的零件数量
        else:
            # 如果当前框放不下那么多零件
            remaining_capacity = self.max_capacity - self.current_capacity
            if remaining_capacity > 0:
                self.current_capacity += remaining_capacity
                num_parts -= remaining_capacity
                return remaining_capacity  # 返回已成功放入框中的零件数量
            return 0  # 返回已成功放入框中的零件数量

    def is_full(self):
        """检查料框是否已满"""
        return self.current_capacity >= self.max_capacity

    def clear(self):
        """清空料框"""
        self.current_capacity = 0

    def __repr__(self):
        return f"Box(size={self.size}, max_capacity={self.max_capacity}, current_capacity={self.current_capacity})"


class Workstation:
    def __init__(self, change_time=None, boxes=None, length=None, width=None, max_capacity=None):
        self.boxes = boxes if boxes else {'large': [], 'medium': [], 'small': []}  # 当前使用中的料框
        # self.empty_boxes = {'large': [], 'medium': [], 'small': []}  # 备用空料框
        # self.filled_boxes = {'large': [], 'medium': [], 'small': []}  # 已满的料框（待入库）
        self.box_length=length if length else {'large': 6000, 'medium': 2500, 'small':1200}
        self.box_width = width if width else {'large': 2500, 'medium': 1500, 'small': 1000}
        self.max_capacity=max_capacity if max_capacity else {'large': 3000, 'medium': 2300, 'small':1300}
        self.change_time = change_time if change_time else {'large': 180, 'medium': 180, 'small':180}  # 换框时间

    def add_boxes(self, size, num_boxes, length, width, max_capacity):
        """初始化工位上的料框"""
        for _ in range(num_boxes):
            self.boxes[size].append(Box(size, length, width, max_capacity))
        # # 先从空料框区拿出若干个填充到 active_boxes 中
        # for _ in range(min(num_boxes, 2)):  # 假设每种Size工位最多同时用2个料框
        #     if self.boxes[size]:
        #         self.boxes[size].append(self.boxes[size].pop())
    #
    def get_active_box(self, size):
        """获取一个未满的料框"""
        for i, box in enumerate(self.boxes[size]):
            if not box.is_full():
                return i, box  # 返回索引和料框
        return None, None  # 当前没有未满的料框，需要换框

    def add_parts(self, size, num_parts, part_type):
        """向指定尺寸的料框中添加零件"""
        total_added_parts = 0
        while num_parts > 0:
            index, box = self.get_active_box(size)
            if box:
                added_parts = box.add_parts(num_parts, part_type)
                total_added_parts += added_parts
                num_parts -= added_parts  # 更新剩余需要添加的零件数量

            if num_parts > 0 and self.all_boxes_full(size):
                # 如果所有框都已满，执行换框
                self.change_box(size,index)
        return f"Successfully added parts: {total_added_parts} parts of type {part_type}."

    def all_boxes_full(self, size):
        """检查所有框是否都已满"""
        return all(box.is_full() for box in self.boxes[size])


    def change_box(self, size):
        """执行换框：满料框移除，加入新的空料框"""
        # 只有当所有框都已满时才换框
        if self.all_boxes_full(size):
            self.boxes[size].append(Box(size, self.box_length[size], self.box_width[size], self.max_capacity[size]))
            return f"All {size} boxes are full. Switched to a new {size} box."
        else:
            return f"Not all {size} boxes are full. No new box needed."

def load_buffer(buffer_name,num_opes_ma):
    if buffer_name == '恒力利材线':
        boxes={'small':[]}
        length = {'small': 1200}
        width = {'small': 1000}
        max_capacity = {'small': 1300}
        change_time={'small':90}
        num_box={'small':18}
    elif buffer_name == '大连重工':
        boxes={'large': [], 'medium': [], 'small': []}
        length = {'large': 6000, 'medium': 2500, 'small':1200}
        width = {'large': 2500, 'medium': 1500, 'small': 1000}
        max_capacity = {'large': 3000, 'medium': 2300, 'small':1300}
        change_time = {'large': 180, 'medium': 180, 'small':180}
        num_box = {'large': 2, 'medium': 4, 'small':18}
    else:
        boxes,length,width,max_capacity,change_time=None,None,None,None,None
        num_box={'large': 2, 'medium': 4, 'small':18}

    workstation = Workstation(change_time=change_time, boxes=boxes,
                                  length=length, width=width, max_capacity=max_capacity)
    for size in num_box:
        workstation.add_boxes(size=size, num_boxes=num_box[size],
                              length=length[size], width=width[size], max_capacity=max_capacity[size])

    # print(workstation.add_parts('large', 80))  # 先填充
    # print(workstation.add_parts('large', 30))  # 触发换框
    print(workstation.add_parts('small', 2000,part_type='a'))
    print(workstation.add_parts('small', 2000, part_type='b'))

    return workstation


def load_buffer1(buffer_name,num_opes_ma):
    num_fenjian=1
    num_small_buffer=0
    num_medium_buffer=0
    num_large_buffer=0

    if buffer_name =='恒力利材线':
        num_fenjian=int(num_opes_ma[0][6])  ##3
        num_small_buffer=18
        buffer_length=1200
        buffer_width = 1000
        max_weight=1300
        change_time=90
    elif buffer_name =='大连重工':
        num_small_buffer = 18
        buffer_length = 1200
        buffer_width = 1000
        max_weight = 1300
        change_time = 180


        # print("num_of_fenjian_gongwei=",num_fenjian)

def nums_extract(lines, layout):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for cur_plate in lines['plate_list']:
        num_opes += len(layout['Operation'])
    num_jobs = len(lines['plate_list'])
    num_mas = len(layout['Nodes'])
    
    return num_jobs, num_mas, num_opes


def edge_extract(cur_plate, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul, layout_config):
    '''
    param:
        cur_plate: 钢板信息
        matrix_proc_time: 每个钢板的工艺处理时间矩阵
        layout_config: 产线布局
    '''
    process_operations = layout_config['Operation'].copy()
    Graph = layout_config['Nodes'].copy()
    block_size = process_operations

    for idx_ope, operation in enumerate(process_operations):
        if idx_ope != len(process_operations) - 1:
            matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
        if idx_ope != 0:
            matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1] \
                     + (torch.arange(matrix_cal_cumul.size(0)) == (idx_ope + num_ope_bias - 1)).to(torch.int8)
        if operation['operation_name'] == '钢板切割':
            alternative_machine = operation['altnative_machine']
            for machine_name in alternative_machine:
                machine = [item for item in Graph if item.get("station_id") == machine_name][0]
                fixed_coef =  machine["fixed_coef"] if "fixed_coef" in machine else 0   # 固定时长
                variable_coefs =  machine["variable_coefs"] if "variable_coefs" in machine else []
                # print("variable_coefs=",variable_coefs)
                nums =  machine["nums"] if "nums" in machine else []
                variable_coef_keys = machine["variable_coef_keys"] if "variable_coef_keys" in machine else []
                # print("variable_coef_keys=",variable_coef_keys) ##['plate_thickness']
                paras = machine["paras"] if "paras" in machine else []
                dur = fixed_coef
                # print("dur=",dur)  ##180
                if not machine['clock']:
                    for idx, variable_coef in enumerate(variable_coefs):
                        key = cur_plate.get(variable_coef_keys[idx])
                        # print("key=",key) ##plate_thickness: 10/10.5/12.5/ ...  4(qiegeji) x 20(gangban)
                        if str(int(key)) in variable_coef:
                            variable_coef = variable_coef[str(int(key))]
                            num = cur_plate.get(nums[idx])
                            dur += ((num/variable_coef/paras[idx]) * 60)
                        elif str(int(key)) not in variable_coef and dur == 0:
                            dur += float('nan')
                    if not math.isnan(dur): 
                        matrix_proc_time[idx_ope+num_ope_bias][machine['station_no']] = dur
        else:
            alternative_machine = operation['altnative_machine']
            for machine_name in alternative_machine:
                machine = [item for item in Graph if item.get("station_id") == machine_name][0]
                if not machine['clock']:   # 判断当前工位是否禁用
                    fixed_coef =  machine["fixed_coef"] if "fixed_coef" in machine else 0  # 固定时长
                    # print("fixed_coef=",fixed_coef) ## 180 for shusong... / 120 for shangliao ...
                    variable_coefs =  machine["variable_coefs"] if "variable_coefs" in machine else []
                    # print("variable_coefs=",variable_coefs)  ## ##[] for others, [25] for xiaojian fenjian
                    nums =  machine["nums"] if "nums" in machine else []
                    # print("nums=",nums)  ##[] for others, ['small_num'] for xiaojian fenjian
                    variable_coef_keys = machine["variable_coef_keys"] if "variable_coef_keys" in  machine else []
                    paras = machine["paras"] if "paras" in machine else []
                    # print("paras=",paras)   ##paras= [1.8] for xiaojianfenjian (conveyor speed)
                    dur = fixed_coef
                    for idx, variable_coef in enumerate(variable_coefs):
                        num = cur_plate.get(nums[idx])
                        # print("num=",num)  ##only small_num that count
                        # print("variable_coef/paras[idx]=",variable_coef/paras[idx])
                        dur += num * variable_coef/paras[idx]
                    if not math.isnan(dur): 
                        matrix_proc_time[idx_ope+num_ope_bias][machine['station_no']] = dur

    return len(process_operations)


def graph_extract(lines, layout_config, total_num_machine, total_num_operation):
    flag = 1
    matrix_proc_time = torch.zeros(size=(total_num_operation, total_num_machine))   # 工序加工时间
    matrix_pre_proc = torch.full(size=(total_num_operation, total_num_operation), dtype=torch.bool, fill_value=False)  # 表示工序之间是否存在前置约束关系（即一个工序必须在另一个工序完成后才能执行）
    matrix_cal_cumul = torch.zeros(size=(total_num_operation, total_num_operation), dtype=torch.int8)
    nums_ope = []  # # 每个工件的工序数量列表
    opes_appertain = np.array([])  # 工序所属的工件索引数组
    num_ope_biases = []  # 每个工件的第一道工序索引
    for cur_plate in lines['plate_list']:
        num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
        num_ope_biases.append(num_ope_bias)
        # print("cur_plate=",cur_plate)
        # Detect information of this job and return the number of operations
        num_ope = edge_extract(cur_plate, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul, layout_config)
        nums_ope.append(num_ope)
        opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
        flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)  # 工序-机器关联矩阵
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(total_num_operation-opes_appertain.size)))

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul

def all_plate_all_part_type(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 用于存放所有 unique 的 part_attri
    unique_part_attri = set()

    # 遍历所有 plate
    for plate in data["plate_list"]:
        # 遍历该 plate 的所有零件
        for part in plate["part_list"]:
            if part["part_type"] == "small":  # 只处理 small 类型的零件
                unique_part_attri.add(part["part_attri"])  # 将 part_attri 加入集合中

    # 转换为列表，方便查看和统计数量
    unique_part_attri_list = list(unique_part_attri)
    return unique_part_attri_list

def every_plate_all_part_type(input_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    # 用于存放每个钢板的零件种类
    plates_part_types = []

    # 遍历每个钢板的数据
    for plate in data["plate_list"]:
        # 用于存放该钢板的所有零件种类
        plate_types = set()

        # 遍历该钢板的零件
        for part in plate["part_list"]:
            if part["part_type"] == "small":  # 只处理 small 类型的零件
                plate_types.add(part["part_attri"])  # 将该零件的种类添加到集合中

        # 将钢板的所有种类转换为列表，并加入到 plates_part_types 中
        plates_part_types.append(list(plate_types))
    return plates_part_types


def build_one_hot(all_plate_part_type, every_plate_part_type, max_part_type_per_batch):
    """
    all_plate_part_type: list[batch][part_types] 每个batch的零件全集
    every_plate_part_type: list[batch][plate][part_types] 每个batch下每块钢板的零件种类
    max_len: 固定 one-hot 编码长度
    """
    one_hot_encodings = []

    for b, (all_types, plates) in enumerate(zip(all_plate_part_type, every_plate_part_type)):
        # 当前 batch 的全集
        batch_types = list(all_types)
        # 如果全集小于 max_len，则补充 None
        if len(batch_types) < max_part_type_per_batch:
            batch_types.extend([None] * (max_part_type_per_batch - len(batch_types)))

        # 存放该 batch 下所有钢板的编码
        batch_encoding = []

        for plate_types in plates:
            encoding = np.zeros(max_part_type_per_batch)
            for part_type in plate_types:
                if part_type in batch_types:
                    idx = batch_types.index(part_type)
                    encoding[idx] = 1
            batch_encoding.append(encoding.tolist())

        one_hot_encodings.append(batch_encoding)

    return one_hot_encodings

def compute_augmented_feat_opes(feat_opes_batch, nums_ope_batch, one_hot_encodings, max_part_type_per_batch):
    """
    feat_opes_batch: [B, F, max_len]         原始工序特征
    nums_ope_batch:  [B, num_jobs]           每个 job 的工序数
    all_plate_part_type: List[List[List]]    每个 batch 中 job 的零件类型
    """
    B, F, max_len = feat_opes_batch.shape
    padded_aug_feats = []

    for b in range(B):
        nums_opes = nums_ope_batch[b]                      # shape: [num_jobs]
        one_hot_encodings=torch.tensor(one_hot_encodings)
        job_onehots = one_hot_encodings[b]             # [num_jobs, 10]

        ope_onehots = []
        # print("nums_opes=",nums_opes)
        for j, num in enumerate(nums_opes.tolist()):
            repeated = job_onehots[j].unsqueeze(0).repeat(num, 1)  # [num, 10]
            ope_onehots.append(repeated)

        ope_feats = torch.cat(ope_onehots, dim=0)          # [N_ope, 10]

        # Padding到 max_len
        pad_len = max_len - ope_feats.size(0)
        if pad_len > 0:
            pad = torch.zeros(pad_len, max_part_type_per_batch)
            ope_feats = torch.cat([ope_feats, pad], dim=0)  # [max_len, 10]

        padded_aug_feats.append(ope_feats.T)                # [10, max_len]

    # 堆叠成 [B, 10, max_len]
    job_onehot_feats = torch.stack(padded_aug_feats, dim=0)

    # 拼接到原始 feat_opes_batch： [B, 6+10, max_len]
    # feat_opes_aug = torch.cat([feat_opes_batch, job_onehot_feats], dim=1)
    feat_opes_batch[:,7:max_part_type_per_batch+7,:]=job_onehot_feats
    return feat_opes_batch


def count_box_switches(every_plate_part_type, box_current_type_batch,buffer_size=6):
    """
        计算每个 batch 中每块钢板在不同工作站放置时的换框预估次数。

        返回一个形状为 [batch_size, num_plates, num_workstations] 的数组
        """
    # 初始化输出列表
    batch_results = []

    for batch_idx, plates in enumerate(every_plate_part_type):
        ws_list = box_current_type_batch[batch_idx]
        batch_switch = []

        for plate_types in plates:
            plate_switch = []
            unique_plate_types = set(plate_types)  # 当前钢板的零件类型集合

            for workstation in ws_list:
                # 如果该工作站完全为空
                if not workstation or all(b is None or b == [] for b in workstation):
                    if len(unique_plate_types) <= buffer_size:
                        plate_switch.append(0)
                    else:
                        plate_switch.append(len(unique_plate_types) - buffer_size)
                    continue

                # 否则，进入正常逻辑
                ws_types = [b for b in workstation if b not in (None, [])]
                existing = unique_plate_types & set(ws_types)
                new_types = unique_plate_types - existing
                empty_slots = buffer_size - len(ws_types)

                if len(new_types) <= empty_slots:
                    plate_switch.append(0)
                else:
                    plate_switch.append(len(new_types) - empty_slots)

            batch_switch.append(plate_switch)
        batch_results.append(batch_switch)

    return np.array(batch_results)

# # 示例数据
# every_plate_part_type = [
#     [['small_G2_806'], ['small_H1G1_201', 'small_G1_201'], ['small_H1G1_206', 'small_G1_206'],
#      ['small_R1G2_112', 'small_G2_112'], ['small_R1G1_525', 'small_G1_525', 'small_H1H1_525', 'small_H1G1_525'], [], [],
#      []],
#     [['small_R1G1_527'], ['small_H1G1_209', 'small_G1_209'], ['small_R1G2_113', 'small_S5G2_603'], [], [],
#      ['small_H1G1_201']]
# ]
#
# self_box_current_type_batch = [
#     [
#         [None, None, None, None, None, None],
#         ['small_G1_209', 'small_H1G1_209', None, None, None, None],
#         [None, None, None, None, None, None]
#     ],
#     [
#         [None, None, None, None, None, None],
#         ['small_R1G2_113', 'small_S5G2_603', None, None, None, None],
#         ['small_H1G1_209', 'small_G1_209', None, None, None, None]
#     ]
# ]
#
# # 计算换框次数
# swap_counts = count_frame_swaps(every_plate_part_type, self_box_current_type_batch)
#
# # 输出结果
# print(f"每个 batch 的换框次数: {swap_counts}")

