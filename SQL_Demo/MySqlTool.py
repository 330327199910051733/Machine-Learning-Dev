import pandas as pd
import os
import mysql.connector
import pymysql
import csv
import json
from abc import abstractmethod, ABCMeta
import time
import datetime


class BaseOperator(metaclass=ABCMeta):
    # @abstractmethod
    # def new(self):
    #     raise NotImplementedError("new is not implemented.")

    @abstractmethod
    def run(self):
        raise NotImplementedError("new is not implemented.")


# 将CSV文件导入到mysql数据库中并执行
def import_csv_to_mysql(csv_file_path, config):
    """
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
    create_table_query = f"CREATE TABLE `{table_name}` ({', '.join([f'`{col}` VARCHAR(100) CHARACTER SET utf8mb4' for col in data.columns])})"
    # print(create_table_query)
    cursor.execute(create_table_query)

    conn.commit()

    # load_data_query = f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\r\n';"
    # # load_data_query = f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\r\n' IGNORE 10 ROWS;"
    # cursor.execute(load_data_query)
    # load_data_query = f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` CHARACTER SET utf8mb4 FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\r\n' IGNORE 1 ROWS;"
    # terminated = "\r\n"
    # sql = f"""
    #     LOAD DATA INFILE '{csv_file_path}'
    #     INTO TABLE {table_name}
    #     CHARACTER SET utf8
    #     FIELDS TERMINATED BY ','
    #     ENCLOSED BY '"'
    #     LINES TERMINATED BY '{terminated}'
    #     IGNORE 1 ROWS
    # """
    # cursor.execute(sql)
    # conn.commit()
    # cursor.execute(load_data_query)
    # 使用 LOAD DATA INFILE 语句将数据导入到 MySQL 数据库中
    # data = pd.read_csv(csv_file_path, sep=',', encoding='utf-8', dtype={'col': str})
    # print(data)
    load_data_query = f"""
        LOAD DATA LOCAL INFILE '{csv_file_path}'
        INTO TABLE `{table_name}`
        CHARACTER SET utf8mb4
        FIELDS TERMINATED BY ','
        ENCLOSED BY '"'
        LINES TERMINATED BY '\r\n'
        IGNORE 1 ROWS
    """
    cursor.execute(load_data_query)
    # 提交更改并关闭游标和连接
    conn.commit()
    cursor.close()
    conn.close()


# 自定义sql类
class MysqlSave(BaseOperator):
    def __init__(self, host: str, user: str, passwd:str, database:str):
        self.content = pymysql.connect(
            host=host,  # mysql的主机ip
            user=user,  # 用户名
            passwd=passwd,  # 数据库密码
            charset='utf8',  # 使用字符集
            database=database,  # 数据库名称

        )
        self.cursor = self.content.cursor()

    def run(self, sql, output_paths):
        """
        导出为csv的函数
        :param sql: 要执行的mysql指令，多个SQL语句用;分隔
        :param output_paths: 导出的csv文件路径列表
        :return:
        """
        # 分割SQL语句
        sql_statements = sql.split(';')

        # 遍历所有SQL语句
        for i, statement in enumerate(sql_statements):
            # 去掉前后空格，转换为大写字母
            statement = statement.strip().upper()

            # 如果语句是SELECT语句，则执行它
            if statement.startswith('SELECT'):
                self.cursor.execute(statement.lower())

                # 拿到表头
                des = self.cursor.description
                title = [each[0] for each in des]

                # 拿到数据库查询的内容
                result_list = []
                for each in self.cursor.fetchall():
                    result_list.append(list(each))

                # 将结果写入CSV文件
                with open(output_paths[i], 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(title)
                    writer.writerows(result_list)

        # 关闭游标和连接
        self.cursor.close()
        self.content.close()
