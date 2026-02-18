import torch
import numpy as np
import math
import json
import copy

class Box:
    def __init__(self, ws_id,size, length, width, max_capacity, type=None):
        self.workstation_id=ws_id
        self.size = size  # 料框尺寸：'large', 'medium', 'small'
        self.length = length  # 长度
        self.width = width  # 宽度
        self.max_capacity = max_capacity  # 最大容量
        self.current_capacity = 0  # 当前存放的零件总质量
        self.current_part_type = type  # 当前存放的零件Type (同类零件才可以放到一个框里, init as None)
        self.is_being_switched = False  # 是否正在进行换框操作
        self.switch_start_time=np.inf ### wu qiong da, no switch
        self.remain_switch_time=np.inf
        self.switch_times=0   ### record the number of change box


    def add_parts(self, num_parts, part_type,part_weight, add_time,current_time):
        """向料框中添加零件"""
        if self.is_being_switched:
            return 0  # 如果框正在换框，则不能添加零件
        if part_type == self.current_part_type or self.current_part_type is None:
            if self.current_capacity + num_parts*part_weight <= self.max_capacity:
                self.current_capacity += num_parts*part_weight
                self.current_part_type = part_type  # 原来是个空框，None变part_type
                return num_parts, add_time * num_parts  # 返回成功添加的零件数量以及花费的时间
            else:
                # 如果当前框放不下那么多零件

                remaining_capacity = self.max_capacity - self.current_capacity
                # print("self.max_capacity=",self.max_capacity)
                # print("remaining_capacity=", remaining_capacity)
                if remaining_capacity >= 0:
                    ### the current box is full, trigger switch
                    self.current_capacity += remaining_capacity
                    self.current_part_type = part_type
                    self.start_switching(current_time)  ###trigger switch at current_time
                    self.switch_times+=1
                    num_parts -= remaining_capacity // part_weight  ##zhengchu
                    # print("remaining_capacity // part_weight=",remaining_capacity // part_weight)
                    return int(remaining_capacity // part_weight), add_time * (remaining_capacity // part_weight)
                    # 返回已成功放入框中的零件数量及花费的时
        return 0, 0  # 返回已成功放入框中的零件数量和时间

    def is_full(self):
        """检查料框是否已满"""
        return self.current_capacity >= self.max_capacity

    def is_empty(self):
        """检查料框是否为空"""
        return self.current_capacity == 0

    def clear(self):
        """清空料框"""
        self.current_capacity = 0

    def start_switching(self, start_time):
        self.is_being_switched = True
        self.switch_start_time=start_time

        # self.switch_remaining_time = switch_time

    def finish_switching(self):
        self.is_being_switched = False
        self.switch_start_time = np.inf
        self.current_capacity = 0
        self.current_part_type=None
        self.remain_switch_time=np.inf


    def __repr__(self):
        return f"Box(ws_id={self.workstation_id},size={self.size}, max_capacity={self.max_capacity}, " \
               f"current_capacity={self.current_capacity}, part_type={self.current_part_type})"


class Workstation:
    def __init__(self, change_time=None, boxes=None, length=None, width=None,
                 max_capacity=None, part_add_time=None):
        self.boxes = boxes if boxes else {'bigger': [],'large': [], 'medium': [], 'small': []}  # 当前使用中的料框
        self.box_length = length if length else {'bigger':4000, 'large': 6000, 'medium': 2500, 'small': 1200}
        self.box_width = width if width else {'bigger':4000, 'large': 2500, 'medium': 1500, 'small': 1000}
        self.max_capacity = max_capacity if max_capacity else {'bigger':4000,'large': 3000, 'medium': 2300, 'small': 1300}
        self.change_time = change_time if change_time else {'bigger':180,'large': 180, 'medium': 180, 'small': 180}  # 换框时间
        self.part_add_time = part_add_time if part_add_time \
            else {'bigger': 60 / 1.4, 'large': 60 / 1.4, 'medium': 45 / 1.4, 'small': 25 / 1.8}  # 加入零件时间
        self.time_elapsed = 0  # 总时间流逝（包括换框时间）

    def box_state_update(self,size,time):
        for i, box in enumerate(self.boxes[size]):
            if box.is_being_switched: ###should check its switch start time, and judge if it has complete switching
                has_switch_time=time-box.switch_start_time
                if has_switch_time >=self.change_time[size]: ## has already changed
                    box.finish_switching()
                else:
                    box.remain_switch_time=self.change_time[size]-has_switch_time

    def add_boxes(self, ws_id,size, num_boxes, length, width, max_capacity):
        """初始化工位上的料框"""
        for _ in range(num_boxes):
            box=Box(ws_id,size, length, width, max_capacity)
            self.boxes[size].append(box)

    def get_active_box(self, size, part_type=None):
        """获取一个未满且不是正在换框的料框，或者可以接受该part_type的空框"""
        # print("len(self.boxes[size])=",len(self.boxes[size]))
        # print("size=",size)
        for i, box in enumerate(self.boxes[size]):
            if not box.is_being_switched  and \
                    (box.current_capacity==0 or box.current_part_type == part_type):
                return i, box  # 返回索引和料框
        return None, None  # 当前没有未满或匹配的料框

    def ws_add_parts(self, size, num_parts, part_type, part_weight,time):
        """向指定尺寸的料框中添加零件"""
        # print("num_parts=",num_parts)
        # print("part_size=",size) ## all small
        self.time_elapsed=copy.deepcopy(time)  ### workstation.add_parts time= current_time

        ### judge all box state accroding to current_time
        self.box_state_update(size,self.time_elapsed)

        total_added_parts = 0
        # total_time = 0
        while num_parts > 0:
            # print("num_parts=",num_parts)
            index, box = self.get_active_box(size, part_type)  ## check if there is eligible box
            if box:
                # print("box")
                curent_time=copy.deepcopy(self.time_elapsed)
                # print("part_weight=",part_weight)
                added_parts, add_time = box.add_parts(num_parts, part_type, part_weight,
                                                      self.part_add_time[size],curent_time)
                total_added_parts += added_parts
                # print("add_time=",add_time)
                self.time_elapsed += add_time
                # print("added_parts=",added_parts)
                num_parts -= added_parts  # 更新剩余需要添加的零件数量
                # print("num_parts=",num_parts)
            else:  ## no box can use (1: a new type come, that no ), then need to change_box
                # 1. 查找 switch 剩余时间最短的框
                # print("no box can use")
                switch_index, switch_box = self.get_fastest_switching_box(size)
                if switch_box:
                    # print("switch_box")
                    current_time=copy.deepcopy(self.time_elapsed)
                    # print("current_time=",current_time)
                    # print("self.change_time[size]=",self.change_time[size])
                    self.time_elapsed += max(0,self.change_time[size] -\
                                         (current_time-switch_box.switch_start_time))
                    ### 更新全局时间为：看当前时间比switch开始的时间走了多久，用总换框时间 -它，就是还要再加的时间
                    switch_box.finish_switching()
                    # self.box_state_update(size, self.time_elapsed)  ## reset the box states
                    continue
                # 2. 没有正在换的框，就找最满的框进行换框
                most_full_index, most_full_box = self.get_most_full_box(size)
                # print("most_full_box=",most_full_box)
                if most_full_box:
                    # print("most_full_box")
                    most_full_box.start_switching(self.time_elapsed) ## at current time, it start switch
                    most_full_box.switch_times+=1
                    self.time_elapsed += self.change_time[size]
                    most_full_box.finish_switching()
                    # print("most_full_box=",most_full_box)
                    continue
                else:
                    # print("no most full box")
                    break
        # print("self.time_elapsed=",self.time_elapsed)
        return self.time_elapsed
        #
        #
        #             # if num_parts > 0:  # 如果零件还没完全放完
        #     #     result = self.change_box(size, part_type)  # 换框操作
        #     #     print(result)
        # return f"Successfully added parts: {total_added_parts} parts of type {part_type}. " \
        #        f"Time taken: {total_time}s."

    def get_fastest_switching_box(self, size):
        # print("get_fastest_switching_box")
        fastest_index = None
        fastest_box = None
        min_start_time = float('inf')

        for i, box in enumerate(self.boxes[size]):
            if box.is_being_switched and box.switch_start_time < min_start_time:
                fastest_index = i
                fastest_box = box
                min_start_time = box.switch_start_time ## find the earliest switch box
        # print("min_start_time=",min_start_time)
        # print("fastest_box=", fastest_box)
        return fastest_index, fastest_box

    def get_most_full_box(self, size):
        """查找最满的框"""  ##所有的框都没有满，也都没有在被换
        most_full_box = None
        most_full_capacity = -1
        most_full_box_index = None

        for i, box in enumerate(self.boxes[size]):
            if box.is_being_switched:
                # print("box.is_being_switched=", box.is_being_switched)
                continue
            if box.current_capacity > most_full_capacity:
                # print("box.current_capacity=",box.current_capacity)
                most_full_box = box

                most_full_capacity = box.current_capacity
                most_full_box_index = i

        return most_full_box_index, most_full_box



def create_subclass(index):
    class_name = f"Box_{index}"
    # 动态继承 Box 类
    subclass = type(class_name, (Box,), {})
    return subclass

def load_buffer(buffer_name):
    box_classes = {}

    if buffer_name == '恒力利材线':
        boxes={'small':[]}
        length = {'small': 1200}
        width = {'small': 1000}
        max_capacity = {'small': 1300}
        change_time={'small':90}
        num_box={'small':6}  ##总共3个分拣工位，每个工位6个，共18个
        for i in range(1, 4):  # 创建 5 个 Box 子类  hengli 3 xian
            box_classes[f"Box_{i}"] = create_subclass(i)

    elif buffer_name == '大连重工':
        boxes={'large': [], 'medium': [], 'small': []}
        length = {'large': 6000, 'medium': 2500, 'small':1200}
        width = {'large': 2500, 'medium': 1500, 'small': 1000}
        max_capacity = {'large': 3000, 'medium': 2300, 'small':1300}
        change_time = {'large': 180, 'medium': 180, 'small':180}
        num_box = {'large': 2, 'medium': 4, 'small':18}
        box_classes[f"Box_{1}"] = create_subclass(1)
    elif buffer_name == '芜湖船厂':
        boxes = {'large': [], 'small': []}
        length = {'large': 6000, 'small': 6000}
        width = {'large': 1500, 'small': 1500}
        max_capacity = {'large': 2000, 'small': 1000}
        change_time = {'large': 300, 'small': 90}
        num_box = {'large': 8, 'small': 12}
    elif buffer_name == '招商威海':
        boxes = {'bigger': [], 'large': [], 'small': []}
        length = {'bigger': 14200, 'large': 6200, 'small': 3000}
        width = {'bigger': 3000, 'large': 3000, 'small': 1500}
        max_capacity = {'bigger': 4000, 'large': 2500, 'small': 1200}
        change_time = {'bigger': 180, 'large': 180, 'small':180}
        num_box = {'bigger': 7, 'large': 8, 'small':33}
    else:
        boxes,length,width,max_capacity,change_time=None,None,None,None,None
        num_box={'large': 2, 'medium': 4, 'small':18}
        box_classes[f"Box_{1}"] = create_subclass(1)

    # num_fenjian = int(num_opes_ma[0][6])  ## 3 gongwei total
    workstation = Workstation(change_time=change_time, boxes=boxes,
                              length=length, width=width, max_capacity=max_capacity)
    # for size in num_box:
    #     workstation.add_boxes(size=size, num_boxes=num_box[size],
    #                           length=length[size], width=width[size], max_capacity=max_capacity[size])
    # all_workstation=[[] for i in range(num_fenjian)]]
    # all_workstation=[]
    # for i in range(num_fenjian):
    #     workstation = Workstation(change_time=change_time, boxes=boxes,
    #                               length=length, width=width, max_capacity=max_capacity)
    #     for size in num_box:
    #         workstation.add_boxes(size=size, num_boxes=num_box[size],
    #                               length=length[size], width=width[size], max_capacity=max_capacity[size])
    #     all_workstation.append(workstation)
    box_param=copy.deepcopy(boxes),copy.deepcopy(length),copy.deepcopy(width),copy.deepcopy(max_capacity),\
              copy.deepcopy(change_time),copy.deepcopy(num_box)
    return workstation,box_param

def fenjian_ope(buffer_name):

    ope_type={'bigger': 1000, 'large': 1000, 'medium': 1000, 'small': 1000, 'tiny':1000}  ##超大件分拣, 大件分拣, ...
    fenjian_ma={'bigger': [], 'large': [], 'medium': [], 'small': [], 'tiny':[]}
    max_part_type_per_batch = 50
    if buffer_name=='恒力利材线':
        fenjian_ope=6
        ope_type['small']=6  ## can do small part
        total_ope=8
        # fenjian_ma=[12,13,14]
        fenjian_ma['small']=[12,13,14]
        num_ma=16
        max_part_type_per_batch=50

    elif buffer_name == '大连重工':
        fenjian_ope=5
        ope_type['small'] = 5 ## can do small part
        ope_type['large'] = 6 ## can do medium, large part
        ope_type['bigger'] = 7  ## can do bigger part
        total_ope = 9
        # fenjian_ma = [6,7,8]
        fenjian_ma['small'] = [6]
        fenjian_ma['large'] = [7]
        fenjian_ma['bigger'] = [8]
        num_ma=10
    elif buffer_name == '芜湖船厂':
        fenjian_ope=6
        ope_type['tiny'] = 6  ## can do tiny part
        ope_type['small'] = 7  ## can do small part
        ope_type['large'] = 8  ## can do medium, large part
        ope_type['bigger'] = 9  ## can do bigger part
        total_ope = 11
        fenjian_ma['tiny'] = [7]
        fenjian_ma['small'] = [8]
        fenjian_ma['large'] = [9]
        fenjian_ma['bigger'] = [10]
        num_ma=12
    else:   ### 招商威海
        fenjian_ope=6
        ope_type['small'] = 4  ## can do small part
        ope_type['large'] = 5  ## can do medium, large part
        ope_type['bigger'] = 6  ## can do bigger part
        total_ope = 8
        fenjian_ma['small'] = [9]
        fenjian_ma['large'] = [10]
        fenjian_ma['bigger'] = [11]
        num_ma=13

    return ope_type,total_ope,fenjian_ma,num_ma,max_part_type_per_batch


def read_json(path:str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_parts(steel_plate_data=None,jobs=None):

        # data_path="./data/parts_data.csv"
        data_plate_path=steel_plate_data
        all_20_plate=read_json(data_plate_path)
        # print("all_20_plate=",all_20_plate)
        # print("all_20_plate=",all_20_plate)
        plate_0=all_20_plate['plate_list'][jobs]
        # print("plate_0=",plate_0)
        plate_id=plate_0['task_no']
        all_part=plate_0['part_list']
        matreial_id=plate_0['material']
        segment=plate_0['segment_id']
        craft='R'
        parts = []
        parts = all_part
        index = 0  ## part_index  212 total

        target_plate_id = None
        # target_plate_id = 'L22000122A150A612_2'
        # for i in range(len(all_part)):
        #     part_id=i
        #     part_name=all_part[i]['part_code']
        #     product_type=all_part[i]['part_attri']
        #     part_length=all_part[i]['part_length']
        #     part_width=all_part[i]['part_width']
        #     part_thickness=all_part[i]['part_thickness']
        #     weight=all_part[i]['part_weight']
        #     part_num=int(all_part[i]['part_num'])
        #     size_type=all_part[i]['part_type']
        #     if size_type != 'small': continue  ##only want small now
        #     if target_plate_id is None:
        #         target_plate_id = plate_id
        #     part_length = float(part_length)
        #     part_width = float(part_width)
        #     part_thickness = float(part_thickness)
        #     weight = float(weight)
        #     part_num = int(part_num)
        #     part_shape = part_length * part_width
        #
        #     for select_id in range(part_num):
        #         part = Part(index, 1, part_id, part_name, product_type, plate_id, matreial_id, part_length,
        #                     part_width, part_thickness, weight, size_type, segment, craft, select_id)
        #         part_type = part.part_type
        #
        #         if part_type not in self.type_dict:
        #             self.type_dict[part_type] = len(self.type_dict) + 1
        #
        #         part_type = self.type_dict[part_type]
        #
        #         job_reso = {}
        #
        #         op_mch_times = np.zeros((1, 1))
        #         op_mch_times[0, 0] = self.conveyor_time  ## 第0个operation（运输）在第0个机器（传送带）上花20s到
        #
        #         for oid in range(self.op_num):  ##3   job[0]=job[1]=job[2]
        #             job_reso[oid] = {
        #                 # value, is_global
        #                 'type': [part_type, 1],
        #                 'shape': [part_shape, 1],
        #                 'picker1': [1, 1],
        #                 'picker2': [1, 1],
        #             }
        #         part.add_ops(op_mch_times, job_reso)
        #         parts.append(part)
        #         if plate_id not in self.steel_plates:
        #             self.steel_plates[plate_id] = []
        #         self.steel_plates[plate_id].append(part)  ## all part in 1 钢板
        #         index += 1
        #
        # print("len(self.steel_plates)=",len(self.steel_plates))  ##20
        # print("index=",index) ##212 parts
        return parts

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
                        print("key=",key) ##plate_thickness: 10/10.5/12.5/ ...  4(qiegeji) x 20(gangban)
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
                    print("fixed_coef=",fixed_coef) ## 180 for shusong... / 120 for shangliao ...
                    variable_coefs =  machine["variable_coefs"] if "variable_coefs" in machine else []
                    print("variable_coefs=",variable_coefs)  ## ##[] for others, [25] for xiaojian fenjian
                    nums =  machine["nums"] if "nums" in machine else []
                    # print("nums=",nums)  ##[] for others, ['small_num'] for xiaojian fenjian
                    variable_coef_keys = machine["variable_coef_keys"] if "variable_coef_keys" in  machine else []
                    paras = machine["paras"] if "paras" in machine else []
                    # print("paras=",paras)   ##paras= [1.8] for xiaojianfenjian (conveyor speed)
                    dur = fixed_coef
                    for idx, variable_coef in enumerate(variable_coefs):
                        num = cur_plate.get(nums[idx])
                        # print("num=",num)  ##only small_num that count
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
        print("cur_plate=",cur_plate)
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

