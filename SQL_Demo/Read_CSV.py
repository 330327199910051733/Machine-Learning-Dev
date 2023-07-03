import pandas as pd
import mysql.connector
import os

def import_csv_to_mysql(csv_folder_path, config):
    """
    将指定文件夹中的所有CSV文件导入到MySQL数据库中，并生成与CSV文件名相同的表。

    :param csv_folder_path: CSV文件夹路径。
    :param config: MySQL数据库连接配置。
    """
    # 连接到MySQL数据库
    conn = mysql.connector.connect(**config)

    # 获取游标对象
    cursor = conn.cursor()

    # 循环遍历CSV文件夹中的所有CSV文件
    for file_name in os.listdir(csv_folder_path):
        if file_name.endswith('.csv'):
            # 提取CSV文件名（不包含扩展名）
            table_name = os.path.splitext(file_name)[0]

            # 使用pandas库读取CSV文件
            csv_file_path = os.path.join(csv_folder_path, file_name)
            data = pd.read_csv(csv_file_path)

            # 检查表是否存在
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            table_exists = cursor.fetchone() is not None
            if table_exists:
                # 如果表已经存在，则删除它
                cursor.execute(f"DROP TABLE {table_name}")

            # 将数据写入MySQL数据库中的表
            create_table_query = f"CREATE TABLE `{table_name}` ({', '.join([f'`{col}` VARCHAR(255)' for col in data.columns])})"
            cursor.execute(create_table_query)
            for row in data.itertuples(index=False):
                insert_row_query = f"INSERT INTO `{table_name}` VALUES ({','.join(['%s']*len(row))})"
                cursor.execute(insert_row_query, row)

    # 提交更改并关闭游标和连接
    conn.commit()
    cursor.close()
    conn.close()

# 测试demo
if __name__ == '__main__':
    # 设置MySQL数据库连接参数
    config = {
      'user': 'root',
      'password': '123456',
      'host': 'localhost',
      'database': 'ssmm7w6d',
      'raise_on_warnings': True
    }

    # # 创建一个测试用的CSV文件夹，并在其中创建两个CSV文件
    # test_folder_path = 'D:\MPC-ML\machine_learning_python-master\SQL_Demo\\test_csv_folder'
    # os.makedirs(test_folder_path, exist_ok=True)
    # with open(os.path.join(test_folder_path, 'test_file_1.csv'), 'w') as f:
    #     f.write('name,age\nAlice,30\nBob,25\n')
    # with open(os.path.join(test_folder_path, 'test_file_2.csv'), 'w') as f:
    #     f.write('id,score\n1,90\n2,85\n3,95\n')

    test_folder_path='D:\MPC-ML\machine_learning_python-master\SQL_Demo\\test_csv_folder'
    # 将CSV文件夹中的文件导入到MySQL中，并生成与文件名相同的表
    import_csv_to_mysql(test_folder_path, config)

    # 查询MySQL数据库中是否存在与CSV文件名相同的表
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute('SHOW TABLES')
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    conn.close()

    # 断言CSV文件夹中的每个CSV文件都已经生成了与文件名相同的表
    for file_name in os.listdir(test_folder_path):
        if file_name.endswith('.csv'):
            table_name = os.path.splitext(file_name)[0]
            assert table_name in tables, f'Table {table_name} not found in MySQL database'

    # 删除测试用的CSV文件夹和其中的文件
    for file_name in os.listdir(test_folder_path):
        file_path = os.path.join(test_folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(test_folder_path)

    print('All tests passed.')