# -*- coding: utf-8 -*-
import os
import time
import json
import re
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from ipdb import set_trace
import logging
import psutil

font_path = '/media/sdb/zhurongjiang/Data_mining/fronts/SourceHanSansSC-Regular.otf'
my_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = my_font.get_name()

# 全局路径配置
data_dir = "/media/sdb/zhurongjiang/Data_mining/10G_data"
data_name = "10G_data_results"
log_filename = f"logs/{data_name}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
os.makedirs(data_name, exist_ok=True)

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.info

# 模块耗时记录
module_time_record = {}
def check_memory_limit():
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024**3)
    log(f"当前内存使用：{used_gb:.2f} GB")
    return used_gb < 250

def record_time(module_name, start_time):
    duration = time.time() - start_time
    module_time_record[module_name] = module_time_record.get(module_name, []) + [duration]
    log(f"⏱️ {module_name} 本单元运行耗时：{duration:.2f} 秒")

def process_file(filepath, part_id):
    log(f"=== 开始处理文件 {filepath} ===")
    part_start = time.time()
    df = pd.read_parquet(filepath)

    # 国家购买总量分析
    start = time.time()
    log(">>> 国家购买偏好分析单元开始")

    if 'purchase_category' not in df.columns:
        df['purchase_category'] = df['purchase_history'].apply(lambda x: eval(x)['category'])

    # 关键修改：不使用 normalize，使用绝对数量
    country_pref = df.groupby('country')['purchase_category'].value_counts().unstack().fillna(0)
    country_pref.to_csv(f"{data_name}/part_{part_id}_purchase_total_by_country.csv")
    log("各国家用户购买总量：")
    log(country_pref.head())

    # 前10用户量国家的购买总量图
    top_countries = df['country'].value_counts().head(10).index
    for country in top_countries:
        if country in country_pref.index:
            plt.figure(figsize=(10, 6))
            country_pref.loc[country].sort_values(ascending=False).plot(kind='bar', color='cornflowerblue')
            plt.title(f"{country} 用户购买类别分布（购买次数）", fontproperties=my_font)
            plt.ylabel("购买次数", fontproperties=my_font)
            plt.xlabel("消费类别", fontproperties=my_font)
            plt.xticks(rotation=45, fontproperties=my_font)
            plt.tight_layout()
            plt.savefig(f"{data_name}/part_{part_id}_purchase_total_{country}.png")
            plt.close()

    record_time("国家偏好分析", start)

def main():
    log(">>> 批量处理开始")
    my_len = 8 if "10G_data" in data_name else 16
    for k in range(my_len):
        file_list = glob.glob(os.path.join(data_dir, f"*{k}.parquet"))
        if not file_list:
            log(f"⚠️ 未找到匹配文件 *{k}.parquet，跳过")
            continue
        if not check_memory_limit():
            log("🚨 内存使用超过 250GB，停止后续处理。")
            break
        file = file_list[0]
        process_file(file, k + 1)

    # 汇总分析：国家购买总量可视化
    log(">>> 汇总国家用户购买总量可视化开始")
    country_pref_list = []
    for i in range(1, my_len + 1):
        part_path = f"{data_name}/part_{i}_purchase_total_by_country.csv"
        if os.path.exists(part_path):
            part_df = pd.read_csv(part_path, index_col=0)
            country_pref_list.append(part_df)

    if country_pref_list:
        total_country_pref = pd.concat(country_pref_list).groupby(level=0).sum()
        total_country_pref.to_csv(f"{data_name}/summary_country_total_preference.csv")

        top_countries = total_country_pref.sum(axis=1).sort_values(ascending=False).head(10).index
        for country in top_countries:
            if country in total_country_pref.index:
                plt.figure(figsize=(10, 6))
                total_country_pref.loc[country].sort_values(ascending=False).plot(kind='bar', color='seagreen')
                plt.title(f"{country} 用户总体购买分布（总次数）", fontproperties=my_font)
                plt.ylabel("购买次数", fontproperties=my_font)
                plt.xlabel("消费类别", fontproperties=my_font)
                plt.xticks(rotation=45, fontproperties=my_font)
                plt.tight_layout()
                plt.savefig(f"{data_name}/summary_purchase_total_{country}.png")
                plt.close()

        # 热力图展示
        plt.figure(figsize=(16, 10))
        sns.heatmap(total_country_pref.loc[top_countries], cmap="YlGnBu", annot=True, fmt=".0f")
        plt.title("前10国家 - 各类别购买总次数热力图", fontproperties=my_font, fontsize=16)
        plt.xlabel("购买类别", fontproperties=my_font)
        plt.ylabel("国家", fontproperties=my_font)
        plt.tight_layout()
        plt.savefig(f"{data_name}/summary_purchase_heatmap.png")
        plt.close()

        log("✅ 国家用户购买总量可视化完成")

if __name__ == "__main__":
    main()
