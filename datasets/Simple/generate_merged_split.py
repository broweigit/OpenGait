import os
import json

# --- 常量配置 ---

# 包含ID子文件夹的数据集路径
# 请确保此脚本与 'dataset_simplified' 文件夹在同一目录下
# 或者在此处提供绝对路径
DATA_DIR = '/data4/movies_ycw/dataset_merged'

# 输出的JSON文件名
OUTPUT_JSON_PATH = '/home/browei/repo/OpenGait_newest/OpenGait/datasets/Simple/Simple.json'


def generate_merged_split_file(source_dir, output_file):
    """
    扫描指定目录，将其中 'nm' 开头的子文件夹名作为训练集ID，
    纯数字的子文件夹名作为测试集ID，并生成一个JSON划分文件。
    """
    print(f"正在扫描目录: '{source_dir}'...")

    # 检查源目录是否存在
    if not os.path.isdir(source_dir):
        print(f"错误: 目录 '{source_dir}' 不存在。请检查路径是否正确。")
        return

    try:
        # 1. 初始化训练集和测试集的ID列表
        train_ids = []
        test_ids = []

        # 2. 遍历源目录下的所有项目
        for name in os.listdir(source_dir):
            # 确保项目是一个文件夹
            if os.path.isdir(os.path.join(source_dir, name)):
                # 根据命名规则进行分类
                if name.startswith('nm'):
                    train_ids.append(name)
                elif name.isdigit():
                    test_ids.append(name)
        
        # 3. 对两个ID列表分别进行排序，以确保每次生成的json文件内容顺序一致
        train_ids.sort()  # 'nm' ID按字母顺序排序即可
        test_ids.sort(key=int)  # 数字ID需要按数值大小排序

        # 检查是否找到了任何ID
        if not train_ids and not test_ids:
            print(f"警告: 在 '{source_dir}' 中未找到 'nm' 开头或纯数字的ID子文件夹。")
            partition_data = {"TRAIN_SET": [], "TEST_SET": []}
        else:
            print(f"成功找到 {len(train_ids)} 个训练ID (以 'nm' 开头): {train_ids}")
            print(f"成功找到 {len(test_ids)} 个测试ID (纯数字): {test_ids}")
            # 4. 构建所需的字典结构
            partition_data = {
                "TRAIN_SET": train_ids,
                "TEST_SET": test_ids
            }

        # 5. 将字典写入JSON文件
        print(f"正在将ID列表写入到 '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as f:
            # indent=4 会让JSON文件格式化，更易于阅读
            json.dump(partition_data, f, indent=4)

        print(f"文件 '{output_file}' 已成功创建。")

    except Exception as e:
        print(f"在处理过程中发生错误: {e}")


# --- 脚本主入口 ---
if __name__ == "__main__":
    generate_merged_split_file(DATA_DIR, OUTPUT_JSON_PATH)