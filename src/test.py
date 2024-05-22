import yaml


def convert_yaml_to_gbk(input_file_path, output_file_path):
    # 读取原始 YAML 文件
    with open(input_file_path, 'r', encoding='utf-8') as f:  # 假设原始文件是 UTF-8 编码
        data = yaml.load(f, Loader=yaml.FullLoader)

    # 写入新的 GBK 编码的 YAML 文件
    with open(output_file_path, 'w', encoding='gbk') as f:
        yaml.dump(data, f, allow_unicode=True)


# 示例使用
input_file_path = 'C-METHOD.yaml'  # 输入的 YAML 文件路径
output_file_path = 'output_gbk.yaml'  # 输出的 GBK 编码的 YAML 文件路径

convert_yaml_to_gbk(input_file_path, output_file_path)

