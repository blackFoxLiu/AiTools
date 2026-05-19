import os

def read_file(file_path, mode='r', encoding='utf-8', return_lines=False):
    """
    读取文件内容。

    参数:
        file_path (str): 文件路径。
        mode (str): 打开模式，默认 'r'（文本只读）。可使用 'rb' 读取二进制文件。
        encoding (str): 文本模式下的编码，默认 'utf-8'。
        return_lines (bool): 若为 True，返回行列表；否则返回整个字符串/字节数据。

    返回:
        str 或 bytes 或 list: 文件内容。文本模式返回字符串，二进制模式返回字节串，
                              若 return_lines=True 则返回行列表（仅文本模式有效）。

    异常:
        FileNotFoundError: 文件不存在。
        PermissionError: 无权限访问。
        UnicodeDecodeError: 编码错误。
        IOError: 其他I/O错误。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        with open(file_path, mode, encoding=encoding if 'b' not in mode else None) as f:
            if return_lines and 'b' not in mode:
                return f.readlines()
            else:
                return f.read()
    except (PermissionError, UnicodeDecodeError, IOError) as e:
        # 可在此添加更具体的处理，例如重试或记录日志
        raise type(e)(f"读取文件失败 '{file_path}': {e}") from e


# 示例用法
if __name__ == "__main__":
    # 读取整个文本文件
    try:
        content = read_file("example.txt")
        print("文本内容:\n", content)
    except FileNotFoundError:
        print("示例文件不存在，请创建一个 example.txt 测试。")

    # 按行读取文本文件
    try:
        lines = read_file("example.txt", return_lines=True)
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line.rstrip()}")
    except FileNotFoundError:
        pass

    # 读取二进制文件（如图片）
    # data = read_file("image.png", mode='rb')
    # print(f"读取到 {len(data)} 字节")