import json
import random
import os

# 文件路径
name='芜湖船厂' #### 恒力利材线 大连重工 招商威海  芜湖船厂
batch_size=20
test_num=10000
input_file = '/home/zss/data/00_FJSP/25_0812_hgnn_real_data/' \
             'dataset/DataWarehouse-master/数据集/'+name+'/完整钢板零件信息.json'
output_dir = '/home/zss/data/00_FJSP/25_0812_hgnn_real_data/dataset/train_real_data/'\
             +name+'_'+str(batch_size)+'/'

# 读取 JSON 文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取 plate_list
plate_list = data.get("plate_list", [])

# 创建输出目录（如果不存在的话）
os.makedirs(output_dir, exist_ok=True)

# 随机抽取100次，每次生成一个独立的JSON文件
for i in range(test_num):
    random_selection = random.sample(plate_list, batch_size)
    output_data = {"plate_list": random_selection}

    # 每次保存为一个新的 JSON 文件
    output_file = os.path.join(output_dir, f"{name}_{i + 1}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"任务文件已保存至 {output_dir}")
