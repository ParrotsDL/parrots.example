# coding=UTF-8
import yaml
import xlrd
import xlwt
from xlutils.copy import copy
import argparse

parser = argparse.ArgumentParser(description='Information Of Benchmark')
parser.add_argument('--config', default='alphatrion.yaml',
                    type=str, help='path to config file')
parser.add_argument('--times', default=1, type=int,
                    help='times of the results to write') # 多次跑模型，会更新不同次的信息，默认第一次跑出的结果 times=1,第二次 times=2,依次类推
args = parser.parse_args()


def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿

def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("写入数据成功！")

def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()


if "__main__" == __name__:
    filename =args.config
    file = open(filename)
    y = yaml.load(file)
    all_key = y.keys()
    all_index=[]
    msg_list = ['benchmark', 'dailytest', 'weeklytest', 'weeklybenchmark', 'dummydata', 'all']
    for i in all_key:
        for j in msg_list:
            tmp = []
            if j in y[i]:
                if len(y[i][j]['__benchmark_avg_iter_time(s)']) >= args.times + 3:
                    tmp.append(j)
                    tmp.append(str(i))
                    tmp.append(y[i][j]['__benchmark_avg_iter_time(s)'][args.times + 2])
                    tmp.append(y[i][j]['__benchmark_mem_alloc(mb)'][args.times + 2])
                    tmp.append(y[i][j]['__benchmark_mem_cached(mb)'][args.times + 2])
                    tmp.append(y[i][j]['__benchmark_pure_training_time(h)'][args.times + 2])
                    tmp.append(y[i][j]['__benchmark_total_time(h)'][args.times + 2])
                    all_index.append(tmp)

    book_name_xls = args.config+'_information.xls'
    sheet_name_xls = 'information'
    value_title = [["信息类型","模型名", "__benchmark_avg_iter_time(s)", "__benchmark_mem_alloc(mb)", "__benchmark_mem_cached(mb)", "__benchmark_pure_training_time(h)","__benchmark_total_time(h)"]]
    value = all_index
    write_excel_xls(book_name_xls, sheet_name_xls, value_title)
    write_excel_xls_append(book_name_xls, value)
    read_excel_xls(book_name_xls)