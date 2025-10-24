import json
import os
# 文件路径
file1_path = "datasets/MUGE/test_texts.jsonl"
file2_path = "datasets/MUGE/test_predictions.jsonl"
output_path = "datasets/MUGE/merge_3.jsonl"


print(f"当前工作目录: {os.getcwd()}")


print(f"file1 存在: {os.path.exists(file1_path)}")
print(f"file2 存在: {os.path.exists(file2_path)}")

# 检查输出目录是否存在，如果不存在则创建
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    print(f"创建目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# 读取第一个文件并存储到字典中
data1 = {}
with open(file1_path, "r", encoding="utf-8") as f1:
    for line in f1:
        obj = json.loads(line.strip())
        # 存储 text_id 和 text 到字典中
        data1[obj['text_id']] = obj['text']
 
# 读取第二个文件并拼接数据
output_data = []
with open(file2_path, "r", encoding="utf-8") as f2:
    for line in f2:
        obj = json.loads(line.strip())
        query_id = obj['text_id']
        if query_id in data1:
            combined_obj = {
                "query_id": query_id,
                "query_text": data1[query_id],
                "item_ids": obj['image_ids']
            }
            output_data.append(combined_obj)

print(output_data[0])

# 写入到输出文件
with open(output_path, "w", encoding="utf-8") as outfile:
    for item in output_data:
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write("\n")

