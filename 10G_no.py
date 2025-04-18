# -*- coding: utf-8 -*-
import os
import time
import json
import re
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import logging
import psutil
font_path = '/media/sdb/zhurongjiang/Data_mining/fronts/SourceHanSansSC-Regular.otf'
my_font = fm.FontProperties(fname=font_path)

# 全局路径配置
data_dir = "/media/sdb/zhurongjiang/Data_mining/10G_data"
data_name = "10G_data_nofilter_results"
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

# 用于记录各模块耗时
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
    # 缺失值分析
    start = time.time()
    rows_with_missing = df.isnull().any(axis=1).sum()
    log(f"缺失值行数: {rows_with_missing}, 占比: {rows_with_missing / len(df) * 100:.2f}%")
    record_time("缺失值分析", start)

    # 省市合法性过滤
    # start = time.time()
    # with open("pc.json", "r", encoding="utf-8") as f:
    #     province_city_dict_pc = json.load(f)

    # def extract_province_and_suffix(address):
    #     prov_match = re.match(r'^(.{2,10}?(省|市|自治区|特别行政区))', address)
    #     if prov_match:
    #         province = prov_match.group(1)
    #         remaining = address[len(province):]
    #         suffix = remaining[:2] if len(remaining) >= 2 else remaining
    #         return pd.Series([province, suffix])
    #     return pd.Series([None, None])

    # df[['province_extracted', 'city_extracted']] = df['chinese_address'].apply(extract_province_and_suffix)

    # def is_valid_pc(row):
    #     prov, city = row['province_extracted'], row['city_extracted']
    #     if pd.isna(prov) or pd.isna(city):
    #         return False
    #     valid_cities = province_city_dict_pc.get(prov, [])
    #     for valid_city in valid_cities:
    #         if city in valid_city or valid_city in city:
    #             return True
    #     return False

    # df['province_city_match'] = df.apply(is_valid_pc, axis=1)
    # invalid_rows = df[df['province_city_match'] == False]
    # invalid_rows.to_csv(f"{data_name}/part_{part_id}_province_city_invalid.csv", index=False)
    # df = df[df['province_city_match'] == True].copy()
    # record_time("省市合法性过滤", start)

    # 异常值检测
    start = time.time()
    def detect_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return ((series < lower) | (series > upper))

    for col in ['age', 'income', 'credit_score']:
        outliers = detect_outliers(df[col])
        log(f"{col} 异常值数量：{outliers.sum()} 占比：{outliers.mean() * 100:.2f}%")
    record_time("异常值检测", start)

    # 数据清洗
    start = time.time()
    before = len(df)
    df = df[(df['age'] >= 0) & (df['income'] >= 0)]
    after = len(df)
    log(f"清洗记录数：{before - after}, 占比：{(before - after) / (before+0.001):.2%}")
    record_time("数据清洗", start)

    # 年龄统计
    start = time.time()
    df['age'].value_counts().sort_index().to_csv(f"{data_name}/part_{part_id}_age_stats.csv")
    record_time("年龄统计", start)

    # 收入和信用评分
    start = time.time()
    df['income_range'] = pd.cut(df['income'], bins=[0, 100000, 300000, 500000, 700000, 1000000], labels=['0-10w', '10w-30w', '30w-50w', '50w-70w', '70w-100w+'], right=False)
    df['credit_range'] = pd.cut(df['credit_score'], bins=[0, 400, 600, 800, 1000], labels=['低（0-400）', '中低（400-600）', '中高（600-800）', '高（800以上）'], right=False)
    df['income_range'].value_counts().sort_index().to_csv(f"{data_name}/part_{part_id}_income_range.csv")
    df['credit_range'].value_counts().sort_index().to_csv(f"{data_name}/part_{part_id}_credit_range.csv")
    record_time("收入信用评分", start)

    # 国家统计
    start = time.time()
    df['country'].value_counts().to_csv(f"{data_name}/part_{part_id}_country_counts.csv")
    record_time("国家统计", start)

    # 用户画像
    start = time.time()
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], labels=["青年", "中年", "中老年", "老年"])
    df['income_level'] = pd.cut(df['income'], bins=3, labels=["低收入", "中收入", "高收入"])
    df['purchase_category'] = df['purchase_history'].apply(lambda x: eval(x)['category'])

    portrait = df.groupby(['age_group', 'income_level', 'gender'])['purchase_category'].value_counts().unstack().fillna(0)
    portrait['total'] = portrait.sum(axis=1)
    portrait = portrait.div(portrait['total'], axis=0)
    portrait.to_csv(f"{data_name}/part_{part_id}_portrait.csv")
    record_time("用户画像", start)
    
    # 性别年龄购买偏好分析
    start = time.time()
    log(">>> 性别、年龄购买偏好单元开始")

    if 'purchase_category' not in df.columns:
        df['purchase_category'] = df['purchase_history'].apply(lambda x: eval(x)['category'])

    gender_pref = df.groupby('gender')['purchase_category'].value_counts(normalize=True).unstack().fillna(0)
    gender_pref.to_csv(f"{data_name}/part_{part_id}_purchase_preference_by_gender.csv")
    log("各性别用户购买偏好：")
    log(gender_pref)

    plt.figure(figsize=(10, 6))
    gender_pref.T.plot(kind='bar')
    plt.title("性别维度购买偏好", fontproperties=my_font)
    plt.ylabel("比例", fontproperties=my_font)
    plt.xlabel("消费类别", fontproperties=my_font)
    plt.xticks(rotation=45, fontproperties=my_font)
    plt.legend(title='性别', loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{data_name}/part_{part_id}_purchase_preference_by_gender.png")
    plt.close()

    age_pref = df.groupby('age_group')['purchase_category'].value_counts(normalize=True).unstack().fillna(0)
    age_pref.to_csv(f"{data_name}/part_{part_id}_purchase_preference_by_age.csv")
    log("各年龄段用户购买偏好：")
    log(age_pref)

    plt.figure(figsize=(10, 6))
    age_pref.T.plot(kind='bar')
    plt.title("年龄段维度购买偏好", fontproperties=my_font)
    plt.ylabel("比例", fontproperties=my_font)
    plt.xlabel("消费类别", fontproperties=my_font)
    plt.xticks(rotation=45, fontproperties=my_font)
    plt.legend(title='年龄段', loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{data_name}/part_{part_id}_purchase_preference_by_age.png")
    plt.close()

    age_gender_pref = df.groupby(['age_group', 'gender'])['purchase_category'].value_counts(normalize=True).unstack().fillna(0)
    age_gender_pref.to_csv(f"{data_name}/part_{part_id}_purchase_preference_by_age_gender.csv")
    log("年龄+性别组合购买偏好（部分预览）：")
    log(age_gender_pref.head())

    for age_group in df['age_group'].dropna().unique():
        subset = age_gender_pref.loc[age_group]
        plt.figure(figsize=(8, 5))
        subset.T.plot(kind='bar')
        plt.title(f"{age_group}年龄段下性别购买偏好", fontproperties=my_font)
        plt.ylabel("比例", fontproperties=my_font)
        plt.xlabel("消费类别", fontproperties=my_font)
        plt.xticks(rotation=45, fontproperties=my_font)
        plt.legend(title='性别', loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{data_name}/part_{part_id}_purchase_pref_by_age_gender_{age_group}.png")
        plt.close()

    record_time("购买偏好分析", start)

    log(f"✅ 文件 {part_id} 处理完成，总耗时：{time.time() - part_start:.2f} 秒")

def generate_final_visualizations():
    log("📊 正在生成合并图表")
    portrait_files = sorted(glob.glob(f"{data_name}/part_*_portrait.csv"))
    dfs = [pd.read_csv(p, index_col=[0, 1, 2]) for p in portrait_files]
    merged_portrait = pd.concat(dfs)

    summary = merged_portrait.groupby(level=[0, 1, 2]).mean()
    summary.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='viridis')
    plt.title("用户画像 - 各类用户消费类别偏好", fontproperties=my_font)
    plt.xlabel("消费类别占比", fontproperties=my_font)
    plt.ylabel("用户画像群体", fontproperties=my_font)
    plt.xticks(rotation=45, fontproperties=my_font)
    plt.yticks(rotation=0, fontproperties=my_font)
    plt.legend(title='消费类别', loc='upper right')
    
    plt.legend(prop=my_font)
    
    
    plt.tight_layout()
    plt.savefig(f"{data_name}/用户画像_消费类别分布图.png")
    plt.close()
    log("📈 用户画像图表保存成功")

def main():
    log(">>> 批量处理开始")
    if "30G_data" in data_name:
        my_len=16
    else:
        my_len=8
    for k in range(0,my_len):
        file_list = glob.glob(os.path.join(data_dir, f"*{k}.parquet"))
        df = pd.read_parquet(file_list)
        #判断是否有任意一个值为空
        if df.isnull().values.any():
            #统计空值的数量
            null_count = df.isnull().sum().sum()
            log(f"⚠️ 文件 *{k}.parquet 中存在空值，数量：{null_count}，跳过")
            set_trace()
    #     set_trace()
    #     if not file_list:
    #         log(f"⚠️ 未找到匹配文件 *{k}.parquet，跳过")
    #         continue
    #     #获取当前的memory使用情况,如果超过300G，则停止处理
    #     if not check_memory_limit():
    #         log("🚨 内存使用超过 250GB，停止后续处理。")
    #         break
    #     # set_trace()
    #     file = file_list[0]
    #     process_file(file, k+1)


    # log(">>> 所有文件处理完成，模块耗时如下：")
    # summary_path = f"{data_name}/module_times.csv"
    # with open(summary_path, "w", encoding="utf-8") as f:
    #     f.write("module,avg_time(s)\n")
    #     for module, durations in module_time_record.items():
    #         avg_time = sum(durations) / len(durations)
    #         log(f"{module}：{avg_time:.2f} 秒")
    #         f.write(f"{module},{avg_time:.2f}\n")

    generate_final_visualizations()
        # 汇总统计与可视化
    log(">>> 开始汇总统计与可视化")
    import matplotlib.ticker as mtick
    
    # 年龄分布汇总可视化
    age_stats_all = pd.DataFrame()
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_age_stats.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            age_stats_all = age_stats_all.add(df, fill_value=0)
    plt.figure(figsize=(10, 6))
    age_stats_all.plot(kind='bar', legend=False, color='skyblue', fontsize=10)
    plt.title("全体用户年龄分布", fontproperties=my_font, fontsize=14)
    plt.xlabel("年龄", fontproperties=my_font)
    plt.ylabel("人数", fontproperties=my_font)
    plt.xticks(rotation=90)
    plt.legend(prop=my_font)

    plt.tight_layout()
    plt.savefig(f"{data_name}/age_distribution.png")
    plt.close()

    # 收入等级分布汇总可视化
    income_dist = pd.Series(dtype=float)
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_income_range.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if df.shape[1] == 1:
                df = df.iloc[:, 0]  # 变成 Series
            income_dist = income_dist.add(df, fill_value=0)
    income_dist = income_dist.sort_index()
    income_dist.plot(kind='bar', color='salmon')
    plt.title("收入等级分布", fontproperties=my_font, fontsize=14)
    plt.xlabel("收入区间", fontproperties=my_font)
    plt.ylabel("人数", fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.tight_layout()
    plt.savefig(f"{data_name}/income_distribution.png")
    plt.close()

    # 信用等级分布汇总可视化
    credit_dist = pd.Series(dtype=float)
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_credit_range.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if df.shape[1] == 1:
                df = df.iloc[:, 0]  # 变成 Series
            credit_dist = credit_dist.add(df, fill_value=0)
    credit_dist = credit_dist.sort_index()
    credit_dist.plot(kind='bar', color='mediumseagreen')
    plt.title("信用评分等级分布", fontproperties=my_font, fontsize=14)
    plt.xlabel("信用评分区间", fontproperties=my_font)
    plt.ylabel("人数", fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.tight_layout()
    plt.savefig(f"{data_name}/credit_distribution.png")
    plt.close()

    # 国家分布可视化
    country_counts = pd.Series(dtype=float)
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_country_counts.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            if df.shape[1] == 1:
                df = df.iloc[:, 0]  # 变成 Series
            country_counts = country_counts.add(df, fill_value=0)
    country_counts = country_counts.sort_values(ascending=False)
    top_countries = country_counts.head(20)
    top_countries.plot(kind='barh', figsize=(10, 8), color='cornflowerblue')
    plt.title("Top 20 国家用户分布", fontproperties=my_font, fontsize=14)
    plt.xlabel("用户数量", fontproperties=my_font)
    plt.legend(prop=my_font)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{data_name}/country_top20.png")
    plt.close()

    #性别年龄购买偏好可视化
    log(">>> 开始绘制全局购买偏好分布（按性别、年龄、性别+年龄）")

    # --- 性别维度购买偏好 ---
    gender_pref_all = pd.DataFrame()
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_purchase_preference_by_gender.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            gender_pref_all = gender_pref_all.add(df, fill_value=0)
    if not gender_pref_all.empty:
        gender_pref_all = gender_pref_all.div(gender_pref_all.sum(axis=1), axis=0)
        gender_pref_all.T.plot(kind='bar', figsize=(10, 6))
        plt.title("性别维度购买偏好（全体用户）", fontproperties=my_font)
        plt.ylabel("比例", fontproperties=my_font)
        plt.xlabel("消费类别", fontproperties=my_font)
        plt.xticks(rotation=45, fontproperties=my_font)
        plt.legend(title='性别', loc='upper right')
        plt.legend(prop=my_font)
        plt.tight_layout()
        plt.savefig(f"{data_name}/all_purchase_preference_by_gender.png")
        plt.close()

    # --- 年龄维度购买偏好 ---
    age_pref_all = pd.DataFrame()
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_purchase_preference_by_age.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            age_pref_all = age_pref_all.add(df, fill_value=0)
    if not age_pref_all.empty:
        age_pref_all = age_pref_all.div(age_pref_all.sum(axis=1), axis=0)
        age_pref_all.T.plot(kind='bar', figsize=(10, 6))
        plt.title("年龄段维度购买偏好（全体用户）", fontproperties=my_font)
        plt.ylabel("比例", fontproperties=my_font)
        plt.xlabel("消费类别", fontproperties=my_font)
        plt.xticks(rotation=45, fontproperties=my_font)
        plt.legend(title='年龄段', loc='upper right')
        plt.legend(prop=my_font)
        plt.tight_layout()
        plt.savefig(f"{data_name}/all_purchase_preference_by_age.png")
        plt.close()

    # --- 年龄+性别维度购买偏好 ---
    age_gender_pref_all = pd.DataFrame()
    for part_id in range(1, my_len+1):
        path = f"{data_name}/part_{part_id}_purchase_preference_by_age_gender.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.set_index(['age_group', 'gender'], inplace=True)
            df.index.names = ['age_group', 'gender']
        if age_gender_pref_all.empty:
            age_gender_pref_all = df.copy()
        else:
            age_gender_pref_all = age_gender_pref_all.add(df, fill_value=0)
    if not age_gender_pref_all.empty:
        grouped = age_gender_pref_all.groupby(level=0)
        for age_group, sub_df in grouped:
            sub_df = sub_df.div(sub_df.sum(axis=1), axis=0)
            sub_df.T.plot(kind='bar', figsize=(10, 6))
            plt.title(f"{age_group} 年龄段下性别维度购买偏好（全体）", fontproperties=my_font)
            plt.ylabel("比例", fontproperties=my_font)
            plt.xlabel("消费类别", fontproperties=my_font)
            plt.xticks(rotation=45, fontproperties=my_font)
            plt.legend(title='性别', loc='upper right')
            plt.legend(prop=my_font)
            plt.tight_layout()
            plt.savefig(f"{data_name}/all_purchase_pref_by_age_gender_{age_group}.png")
            plt.close()
    
    
    log("✅ 汇总图表绘制完成")


if __name__ == "__main__":
    main()
