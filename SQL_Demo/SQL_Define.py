import pandas as pd
import mysql.connector
import os
import pymysql
import csv

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

#自定义sql
class MysqlSave:

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

    def search_and_save(self, sql, csv_file):
        """
        导出为csv的函数
        :param sql: 要执行的mysql指令，多个SQL语句用;分隔
        :param csv_file: 导出的csv文件名
        :return:
        """
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
    # for file_name in os.listdir(test_folder_path):
    #     file_path = os.path.join(test_folder_path, file_name)
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)
    # os.rmdir(test_folder_path)

    print('All tests passed.')

    #进行自定义sql处理
    mysql = MysqlSave()
    mysql.search_and_save(
        "select * from house_prices1 where area>=100;select * from house_prices1 where rooms=2;select * from house_prices1 where age>5;select*from person where age>=35;select * from demo where Gender='Male';",
        'test.csv')