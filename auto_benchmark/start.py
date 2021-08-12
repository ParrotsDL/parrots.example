import os
import sys
from auto_benchmark.utils.db_utils import (build_fail_sh,
                                           write_to_xls,
                                           get_data_from_db,
                                           from_dh_sort_table,
                                           add_new_data_to_old_excel,
                                           to_target_format)

framelist = os.path.dirname(__file__) + "/benchmark_model_list.yaml"
partition = sys.argv[1]

# at most 3 times
for time in range(3):
    if time == 0:
        os.system(f'sh auto_benchmark/merged_{time}.sh {partition}')

    # get data from db
    print(f"------------------Get data from db {time} time-----------------")
    get_data_from_db(f"auto_benchmark/data_from_db_{time}.xls")

    # build_fail_sh
    build_fail_sh(framelist=framelist,
                  datasource=f"auto_benchmark/data_from_db_{time}.xls",
                  shsource=f"auto_benchmark/merged_{time}.sh",
                  shtarget=f"auto_benchmark/merged_{time+1}.sh")

    # run
    print(f"*****************run merged_{time+1}.sh************************")
    os.system(f"sh auto_benchmark/merged_{time+1}.sh {partition}")
    if time == 2:
        print("Done 3 times benchmark")

print("*****************build last sh************************")
get_data_from_db("auto_benchmark/data_from_db_last.xls")
build_fail_sh(framelist=framelist,
              datasource="auto_benchmark/data_from_db_last.xls",
              shsource="auto_benchmark/merged_3.sh",
              shtarget="auto_benchmark/last_fail.sh")
# build target table
# from db to sorted table according to frame lavel
# source_file：从db拉下来的数据；save_path: 目标数据保存路径
# 将source_file数据精简；
# 选出列为'模型序号', '框架', '测试方式', '模型', '卡数', '版本', 's/iter'
write_to_xls(framelist, source_file="auto_benchmark/data_from_db_last.xls",
             save_path="auto_benchmark/sorted_data_from_db_.xls")

## （可选,不必要）每个模型dummy和real放到一起，以模型粒度
# to_target_format(framelist,
#                  source="auto_benchmark/pat_1.2real+dummy_.xls",
#                  target="auto_benchmark/pat_1.2real+dummy_+.xls")

## 将dh原始表格按模型优先级排序处理
# source_file: 原始乱序表格，save_path: 目标数据保存路径

# 该操作只需要做一次，后面就用这次生成的表格即可。

# from_dh_sort_table(framelist,
#                    source_file="auto_benchmark/sorted_db_data.xls",
#                    save_path="auto_benchmark/dh_origin_table.xls")

#每个模型dummy和real放到一起，以模型粒度
# 每个模型不同配置文件放在一起比较
to_target_format(framelist,
                 source="auto_benchmark/sorted_data_from_db_.xls",
                 target="auto_benchmark/dh_origin_table.xls"
                )

# 将新的benchmark数据填到旧表，分为两列：real和dummy
# old_file: 现有的排序好的表格，该表格已固定
# new_file：从db拉下来的数据
# save_path：用于存储最终表格，建议直接命名为有效信息，如pat0.11rc
add_new_data_to_old_excel(old_file="auto_benchmark/dh_origin_table_0.12.xls",
                          new_file="auto_benchmark/dh_origin_table.xls",
                          save_path="auto_benchmark/target_table.xls")
