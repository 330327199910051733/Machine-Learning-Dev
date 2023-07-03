import json
import os




folder_path = './test_csv_folder'
# 遍历指定目录下的所有 JSON 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)

        # 读取 JSON 文件
        with open(file_path) as f:
            data = json.load(f)

        # 提取 datasetInfo 字段
        dataset_info = data['datasetInfo']

        # 提取 features 数组
        features = dataset_info['features']

        # 遍历 features 数组，拼接文件路径
        for feature in features:
            operator_id = feature['operatorId']
            input_path = feature['input']
            # file_path = os.path.join(operator_id,"\output\\", input_path)
            file_path = "/".join(["./",operator_id, "output", input_path])
            print(file_path)
        filename = os.path.basename(file_path)
        print(filename)
        filename = os.path.basename(file_path)
        PreName = filename.split(".")[0]
        print(PreName)
