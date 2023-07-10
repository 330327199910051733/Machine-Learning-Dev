import os
import time
import json
import datetime
import mysql.connector
from MySqlTool import import_csv_to_mysql, MysqlSave


def main():
    # 设置MySQL数据库连接参数
    config = {
        'user': 'root',
        'password': '123456',
        'host': '10.50.1.95',
        'database': 'mpc_database',
        'raise_on_warnings': True,
        'allow_local_infile':True
    }

    # 记录程序开始时间
    start_time = time.monotonic()

    # test_csv_file_path='./BB15A6463D364FC89E72FB94BD63A9C6/output/alice.csv'
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

            # 提取params
            params = data['params']

            # 提取params的sql语句
            sql = params['sql']
            # print(sql)

            # 遍历 features 数组，拼接文件路径
            for feature in features:
                operator_id = feature['operatorId']
                input_path = feature['input']
                # file_path = os.path.join(operator_id,"\output\\", input_path)
                test_csv_file_path = "/".join(["./", operator_id, "output", input_path])
                # print(test_csv_file_path)
                filename = os.path.basename(file_path)
                base_name = os.path.basename(test_csv_file_path)
                PreName = os.path.splitext(base_name)[0]
                # print(PreName)
                # print(filename)
                # 将指定CSV文件导入到MySQL中，并生成与文件名相同的表
                import_csv_to_mysql(test_csv_file_path, config)

    # 查询MySQL数据库中是否存在与CSV文件名相同的表
    # mysql = MysqlSave('localhost', 'root', '123456', 'mpc_database')
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute('SHOW TABLES')
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    conn.close()

    # 断言CSV文件已经生成了与文件名相同的表
    table_name = os.path.splitext(os.path.basename(test_csv_file_path))[0]
    assert table_name in tables, f'Table {table_name} not found in MySQL database'

    print('All tests passed.')
    # 记录程序结束时间
    end_time = time.monotonic()
    # 计算程序运行时间
    elapsed_time = datetime.timedelta(seconds=end_time - start_time)

    # 输出程序运行时间
    print(f"程序运行时间为：{elapsed_time}")

    # 进行自定义sql处理
    # own_mysql = MysqlSave()
    own_mysql = MysqlSave(
        host=config.get("host"), user=config.get("user"), passwd=config.get("password"), database=config.get("database")
    )

    # query = f"select * from alice where area>=150;select * from alice where rooms=2; select*from demo where age>=25;select * from person where gender='M';"
    query = sql
    queries = query.split(";");
    # 去除每个子查询的前后空格
    queries = [q.strip() for q in queries if q.strip()]
    # 计算 SELECT 语句的数量
    count = sum(q.lower().startswith('select') for q in queries)
    # print(count)

    # 构建输出目录和文件路径
    output_dir = "output"
    output_file = "test.csv"
    output_path = os.path.join(output_dir, output_file)
    print(output_path)

    # 构建输出目录
    output_dir = "output"

    # 生成 CSV 文件名列表
    output_file_names = [f"result{i + 1}.csv" for i in range(count)]

    # 生成输出路径列表
    output_paths = [os.path.join(output_dir, f) for f in output_file_names]
    print(output_file_names)
    print(output_paths)

    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 PreName 变量构造查询字符串,'PreName就是alice去掉了csv'
    # query = f"select * from alice where area>=150;select * from alice where rooms=2; select*from demo where age>=25;select * from person where gender='M';"
    # query=sql
    # print(sql)
    # 执行查询并将结果保存到输出文件
    own_mysql.run(query, output_paths)


if __name__ == '__main__':
    main()
