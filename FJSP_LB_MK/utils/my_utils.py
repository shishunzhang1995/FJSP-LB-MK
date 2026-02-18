import json
import math

def read_json(path:str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def write_json(data:dict, path:str):
    with open(path, 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


def get_plate_info(json_data):
    plates_info = []
    data = json_data['data']
    for i in range(len(data['plate_info'])):
        plates_info.append({'task_no': data['plate_info'][i]['task_no'], 
                           'cut_total_length': data['plate_info'][i]['cut_total_length'],
                           'groove_total_length': data['plate_info'][i]['groove_total_length'], 
                           'groove_thickness': data['plate_info'][i]['groove_thickness'], 
                           'plate_length': data['plate_info'][i]['length'],
                           'plate_width': data['plate_info'][i]['width'], 
                           'plate_thickness': data['plate_info'][i]['thickness'],
                           'put_priority': data['plate_info'][i]['put_priority'],
                           'material': data['plate_info'][i]['material'],
                           'draw_code': data['plate_info'][i]['draw_code'],
                           'part_list': data['plate_info'][i]['part_list']})
        
    return plates_info




