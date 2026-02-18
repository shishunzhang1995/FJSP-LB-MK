import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Agg')
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import pearsonr



# 读取 Excel 文件
def read_excel_column(file_path, sheet_name, column_name):
    # 使用 pandas 读取 Excel 文件中的指定列
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name]


# 计算相关性
def calculate_correlation(data1, data2):
    # 计算 Pearson 相关系数
    correlation, _ = pearsonr(data1, data2)
    return correlation


# 可视化数据的相关性
def plot_correlation(data1, data2):
    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data1, y=data2)

    # 设置图表标题和标签
    plt.title("Scatter Plot: Correlation between Number_of_part_type and Switch_times")
    plt.xlabel("Number_of_part_type")
    plt.ylabel("Switch_times")

    # 显示图表
    plt.savefig("./gantt_result_0416/plot_data/Dalian_Scatter_Plot_{0}".format(str_time))


def plot_with_trendline(data1, data2):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=data1, y=data2, scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'red', 'lw': 2})

    # 设置图表标题和标签
    plt.title("Heng Li production line\n (Jobs=20, Sorting line =3, Buffer number: {small buffer: 18})")
    plt.xlabel("Number of part type")
    plt.ylabel("Switch times")

    # 显示图表
    plt.savefig("./gantt_result_0416/plot_data/Dalian_Scatter_Plot_with_trendline_{0}".format(str_time))


# 绘制热力图 (如果有多个变量，可以用这种方法展示相关性矩阵)
def plot_heatmap(data1, data2, data3):
    # 将两列数据合并为 DataFrame
    df = pd.DataFrame({'part_type': data1, 'part_num': data2,'switch_times': data3})

    # 计算相关性矩阵
    correlation_matrix = df.corr()

    # 绘制热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=1,
                cbar_kws={'shrink': 0.8})

    # 设置标题
    plt.title("Correlation Heatmap")
    plt.savefig("./gantt_result_0416/plot_data/Dalian_Correlation_heatmap_small{0}".format(str_time))


# 主函数
def main():
    # 输入 Excel 文件路径、sheet 名和列名
    ### Heng Li
    # file1 = './save_hengli/test_20250416_213104/part_type_20250416_213104.xlsx'  # 替换为实际的文件路径
    # file2 = './save_hengli/test_20250416_213104/switch_times_20250416_213104.xlsx'  # 替换为实际的文件路径

    # file1 = './save_all_dataset/大连重工_20250417_162806/part_type_20250417_162806.xlsx'  # 替换为实际的文件路径
    # file2 = './save_all_dataset/大连重工_20250417_162806/switch_times_20250417_162806.xlsx'  # 替换为实际的文件路径

    file_part_type = './save_all_dataset/大连重工_20250419_221106/part_type_20250419_221106.xlsx'
    file_part_num = './save_all_dataset/大连重工_20250419_221106/part_num_20250419_221106.xlsx'
    file_switch_times = './save_all_dataset/大连重工_20250419_221106/switch_times_20250419_221106.xlsx'

    part_type={'large': [], 'medium': [], 'small': []}
    sheet_name1 = 'Sheet1'  # 替换为实际的 sheet 名
    sheet_name2 = 'Sheet1'  # 替换为实际的 sheet 名
    column_large = 'large'  # 替换为实际的列名
    column_medium = 'medium'  # 替换为实际的列名
    column_small = 'small'  # 替换为实际的列名


    # 读取数据
    part_type_large = read_excel_column(file_part_type, sheet_name1, column_large)
    part_type_medium = read_excel_column(file_part_type, sheet_name1, column_medium)
    part_type_small = read_excel_column(file_part_type, sheet_name1, column_small)

    part_num_large = read_excel_column(file_part_num, sheet_name1, column_large)
    part_num_medium = read_excel_column(file_part_num, sheet_name1, column_medium)
    part_num_small = read_excel_column(file_part_num, sheet_name1, column_small)

    switch_times_large = read_excel_column(file_switch_times, sheet_name1, column_large)
    switch_times_medium = read_excel_column(file_switch_times, sheet_name1, column_medium)
    switch_times_small = read_excel_column(file_switch_times, sheet_name1, column_small)

    # data2 = read_excel_column(file_part_num, sheet_name2, column_name2)

    # 计算相关性
    # correlation = calculate_correlation(data1, data2)
    # print(f"The Pearson correlation coefficient between the two datasets is: {correlation:.4f}")
    #
    # # 可视化数据
    # plot_correlation(data1, data2)
    #
    # # 可视化数据：趋势线
    # plot_with_trendline(data1, data2)

    # 可视化数据：热力图（如果需要查看多个数据之间的关系）

    plot_heatmap(part_type_small, part_num_small,switch_times_small)


if __name__ == "__main__":
    main()
