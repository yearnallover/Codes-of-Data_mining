import pandas as pd
import os
from ipdb import set_trace

# 读取 CSV 文件（假设文件名为 'age_count.csv'）
data_name = "10G_data_results"
# data_name="10G_data_nofilter_results"
# data_name = "30G_data_results"
# data_name="30G_data_nofilter_results"

# 根据数据集名判断 part 文件数量
my_len = 16 if "30G_data" in data_name else 8

# 初始化总统计表
num=0
# 循环读取每个分片文件
for part_id in range(1, my_len + 1):
    path = f"{data_name}/part_{part_id}_age_stats.csv"
    # set_trace()
    if os.path.exists(path):
        
        df = pd.read_csv(path, index_col=0)
        # 如果第一次读入就直接赋值
        total_count = df['count'].sum()
        num+=total_count
print(f"总计：{num}")

# 转换为整数

# 可选：保存合并后的统计结果
# age_stats_all.to_csv(f"{data_name}_merged_age_stats.csv")

# # 预览前几行
# print(age_stats_all.head())
print(300000000-8106573)
print((300000000-8106573)/300000000)
print((100000000-2710125))
print((100000000-2710125)/100000000)