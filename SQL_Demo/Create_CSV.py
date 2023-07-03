import csv

# 要写入的测试数据
data = [
    ['Name', 'Age', 'Gender', 'City'],
    ['Alice', '25', 'Female', 'New York'],
    ['Bob', '30', 'Male', 'San Francisco'],
    ['Charlie', '35', 'Male', 'Los Angeles'],
    ['David', '40', 'Male', 'Chicago'],
    ['Eve', '', 'Female', 'Miami'],
    ['', '45', 'Male', 'Houston'],
    ['Grace', '50', '', 'Seattle']
]

# 打开 CSV 文件并创建写入器
with open('test_csv_folder/demo.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # 将数据写入 CSV 文件
    writer.writerows(data)

print('CSV file created.')