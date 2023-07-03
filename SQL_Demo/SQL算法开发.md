### 1. 自定义SQL算子

使用 pymysql 连接 MySQL 数据库、执行 SQL 查询语句，并将查询结果导出为 CSV 文件。

需要安装了以下软件和库：

- Python 3.x
- pymysql 库

使用 Python 和 pymysql 将 MySQL 数据导出为 CSV 文件的步骤：

1. 连接 MySQL 数据库,并读取CSV文件导入到mysql里面

使用 pymysql 库连接 MySQL 数据库。在连接之前，需要指定 MySQL 数据库的主机名、用户名、密码和数据库名等参数。如：![image-20230629105632712](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629105632712.png)

![image-20230629105650937](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629105650937.png)

   2.执行 SQL 查询语句并导出为 CSV 文件

使用 pymysql 库执行 SQL 查询语句，并将查询结果导出为 CSV 文件。

具体而言，该方法会：

- 分割 SQL 语句。
- 打开 CSV 文件并创建 CSV 写入器。
- 遍历 SQL 语句，如果语句是 SELECT 语句，则执行它。
- 拿到查询结果的表头和内容，并将结果写入 CSV 文件。
- 关闭游标和连接。

​	3.调用 search_and_save 方法

在调用 `search_and_save` 方法时，需要指定要执行的 SQL 查询语句和导出的 CSV 文件名。

总结：使用 Python 和 pymysql 将 MySQL 数据导出为 CSV 文件的方法。具体而言，使用 pymysql 连接 MySQL 数据库、执行 SQL 查询语句，并将查询结果导出为 CSV 文件。

![image-20230629105816378](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629105816378.png)

![image-20230629111018061](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629111018061.png)



### 2.特征统计

使用 Python 和 Pandas 进行数据分析和 MySQL 数据库操作

先将CSV文件中读入数据并导入到Mysql中然后，从 MySQL 数据库中读取数据、对数据进行分析和处理，并将结果输出到 CSV 文件

需要安装以下软件和库：

- Python 3.x

- MySQL 数据库

- Pandas 库

- mysql-connector-python 库

  

1.导入CSV数据并连接 MySQL 数据库

使用 mysql-connector-python 库连接 MySQL 数据库。在连接之前，需要指定 MySQL 数据库的主机名、用户名、密码和数据库名等参数。

2.从 MySQL 数据库中读取数据

使用 Pandas 库从 MySQL 数据库中读取数据。在读取数据之前，需要指定 SQL 查询语句和数据库连接对象。

3.对特征统计信息进行分析和处理

使用 Pandas 库对数据进行分析和处理。指定特征列的统计信息`(count, mean, std, min, 25%, 50%, 75%, max)`，输出到feature.csv文件中。

4.将结果输出到 CSV 文件中

5.关闭数据库连接

结果大致呈现：

![image-20230629105825054](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629105825054.png)

![image-20230629105833242](C:\Users\86187\AppData\Roaming\Typora\typora-user-images\image-20230629105833242.png)











