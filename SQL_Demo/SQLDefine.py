import pandas as pd
import mysql.connector
import os
import pymysql
import csv
import json
from abc import abstractmethod, ABCMeta


class BaseOperator(metaclass=ABCMeta):

    # @abstractmethod
    # def new(self):
    #     raise NotImplementedError("new is not implemented.")

    @abstractmethod
    def run(self):
        raise NotImplementedError("new is not implemented.")

#将CSV文件导入到mysql数据库中并执行
def import_csv_to_mysql(csv_file_path, config):
    """
    将指定CSV文件导入到MySQL数据库中，并生成与CSV文件名相同的表。

    :param csv_file_path: CSV文件路径。
    :param config: MySQL数据库连接配置。
    """
    # 连接到MySQL数据库
    conn = mysql.connector.connect(**config)

    # 获取游标对象
    cursor = conn.cursor()

    # 提取CSV文件名（不包含扩展名）
    table_name = os.path.splitext(os.path.basename(csv_file_path))[0]

    # 使用pandas库读取CSV文件
    data = pd.read_csv(csv_file_path)
    # 将 NaN 值替换为空字符串
    data.fillna('', inplace=True)
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

#自定义sql类
class MysqlSave(BaseOperator):

    def __init__(self):
        self.content = pymysql.Connect(
            host='localhost',  # mysql的主机ip
            port=3306,  # 端口
            user='root',  # 用户名
            passwd='123456',  # 数据库密码
            charset='utf8',  # 使用字符集
            database='ssmm7w6d'  # 数据库名称
        )
        self.cursor = self.content.cursor()

    # def new(self):
    #     conn = pymysql.Connect(
    #         host='localhost',  # mysql的主机ip
    #         port=3306,  # 端口
    #         user='root',  # 用户名
    #         passwd='123456',  # 数据库密码
    #         charset='utf8',  # 使用字符集
    #         database='ssmm7w6d'  # 数据库名称
    #     )
    #     return conn


    def run(self, sql, csv_file):
        """
        导出为csv的函数
        :param sql: 要执行的mysql指令，多个SQL语句用;分隔
        :param csv_file: 导出的csv文件名
        :return:
        """
        # conn=self.new()
        # cursor=conn.cursor()

        # 分割SQL语句
        sql_statements = sql.split(';')

        # 打开CSV文件并创建写入器
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # 遍历所有SQL语句
            for statement in sql_statements:
                # 去掉前后空格，转换为大写字母
                statement = statement.strip().upper()

                # 如果语句是SELECT语句，则执行它
                if statement.startswith('SELECT'):
                    self.cursor.execute(statement)

                    # 拿到表头
                    des = self.cursor.description
                    title = [each[0] for each in des]

                    # 拿到数据库查询的内容
                    result_list = []
                    for each in self.cursor.fetchall():
                        result_list.append(list(each))

                    # 将结果写入CSV文件
                    writer.writerow(title)
                    writer.writerows(result_list)

        # 关闭游标和连接
        self.cursor.close()
        self.content.close()


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

    #进行自定义sql处理
    mysql = MysqlSave()

    # 构建输出目录和文件路径
    output_dir = "output"
    output_file = "test.csv"
    output_path = os.path.join(output_dir, output_file)

    # 如果输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 PreName 变量构造查询字符串,'PreName就是alice去掉了csv'
    query = f"select * from alice where area>=150;select * from alice where rooms=2; select*from demo where age>=25;select * from person where gender='M'"

    # 执行查询并将结果保存到输出文件
    mysql.run(query, output_path)