import os
import json
import math

# --- 常量配置 ---

# 数据集根目录。脚本会扫描此目录下的第一层子文件夹作为ID。
# 之前脚本的输出 'dataset_simplified' 或 'dataset_simplified_with_id' 都可以作为输入。
DATA_DIR = '/data4/movies_ycw/dataset_merged' 

# 输出的JSON划分文件名
OUTPUT_JSON_PATH = '/home/browei/repo/OpenGait_newest/OpenGait/datasets/Simple/Simple.json'

# --- 核心配置：可在此处修改训练集所占的比例 ---
# 0.7 代表 70% 训练, 30% 测试
# 0.8 代表 80% 训练, 20% 测试
TRAIN_RATIO = 4 / 5 


def create_dataset_partition_file(source_dir, output_file, train_ratio=0.8):
    """
    扫描指定目录，将其中的子文件夹名作为ID，并根据指定的比例划分为训练集和测试集，
    最终生成一个JSON划分文件。

    Args:
        source_dir (str): 数据集根目录。
        output_file (str): 输出的JSON文件路径。
        train_ratio (float): 训练集所占的比例，应在 0.0 到 1.0 之间。
    """
    print(f"正在扫描目录: '{source_dir}'...")

    # 1. 合法性检查
    if not 0.0 <= train_ratio <= 1.0:
        print(f"错误: 训练集比例 (TRAIN_RATIO) '{train_ratio}' 无效，必须在 0.0 和 1.0 之间。")
        return

    if not os.path.isdir(source_dir):
        print(f"错误: 目录 '{source_dir}' 不存在。请检查路径是否正确。")
        return

    try:
        # 2. 读取所有子文件夹名作为ID
        #    (已移除 name.isdigit() 限制，使其能处理任意名称的文件夹)
        id_list = [
            name for name in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, name))
        ]

        # 3. 对ID进行排序，确保每次运行的划分结果都一致
        id_list.sort()
        
        if not id_list:
            print(f"警告: 在 '{source_dir}' 中未找到任何子文件夹。")
            partition_data = {"TRAIN_SET": [], "TEST_SET": []}
        else:
            print(f"成功找到 {len(id_list)} 个ID。")

            # --- 核心修改：按比例划分ID列表 ---
            # a. 计算分割点索引
            split_index = math.ceil(len(id_list) * train_ratio)
            
            # b. 前半部分给训练集
            train_ids = id_list[:split_index]
            
            # c. 后半部分给测试集
            test_ids = id_list[split_index:]
            
            print("-" * 40)
            print(f"划分比例: {train_ratio*100:.1f}% 训练 / {(1-train_ratio)*100:.1f}% 测试")
            print(f"划分结果: {len(train_ids)} 个ID用于训练, {len(test_ids)} 个ID用于测试。")
            # 如果列表太长，可以只打印部分内容
            # print(f"TRAIN_SET (前5个): {train_ids[:5]}")
            # print(f"TEST_SET (前5个): {test_ids[:5]}")
            print("-" * 40)

            # 4. 构建新的字典结构
            partition_data = {
                "TRAIN_SET": train_ids,
                "TEST_SET": test_ids
            }

        # 5. 将字典写入JSON文件
        print(f"正在将ID列表写入到 '{output_file}'...")
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"输出目录 '{output_dir}' 不存在，将自动创建。")
            os.makedirs(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 来正确显示中文等非ASCII字符
            json.dump(partition_data, f, indent=4, ensure_ascii=False)

        print(f"文件 '{output_file}' 已成功创建。")

    except Exception as e:
        print(f"在处理过程中发生错误: {e}")


# --- 脚本主入口 ---
if __name__ == "__main__":
    # 将配置的常量传入函数
    create_dataset_partition_file(DATA_DIR, OUTPUT_JSON_PATH, TRAIN_RATIO)