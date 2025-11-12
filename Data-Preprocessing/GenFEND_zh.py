import time
import json
from openai import OpenAI

# ========== 1. 初始化客户端 ==========
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="xxxxxxxxxx"
)

model_name = "qwen3-235b-a22b-instruct-2507"

start_time = time.time()

# ========== 2. 属性定义 ==========
genders = ["男性", "女性"]
ages = [
    "你的年龄小于17岁",
    "你的年龄在18岁-29岁之间",
    "你的年龄在30岁-49岁之间",
    "你的年龄在50岁-64岁之间",
    "你的年龄在65岁以上",
]
educations = [
    "你的学历为高中及以下",
    "你是未毕业大学生",
    "你的学历为大学本科及以上",
]

# ========== 3. 输入输出路径 ==========
input_path = "./train.json"
output_path = "./train_comments.json"

# ========== 4. 读取输入数据 ==========
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========== 5. 逐条生成评论并写入 ==========
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        news_id = item.get("source_id")
        print(f"\n📰 处理第 {idx+1} 条新闻 ID: {news_id}")

        content = item.get("content", "").strip()
        label = item.get("label", "")
        time_stamp = item.get("time", "")

        if not content:
            print(f"[{idx+1}] 空文本跳过。")
            continue

        comments = []

        # 遍历 30 种身份组合
        for gender in genders:
            for age in ages:
                for edu in educations:
                    system_prompt = (
                        f"假设你是一名{gender}微博用户，你{age}，"
                        f"教育程度是{edu}。现在你将看到一篇微博新闻，请写一条评论。"
                        "评论需要符合你的身份特征，简短自然，像普通微博用户的发言。"
                    )
                    user_prompt = f"微博新闻内容如下：\n{content}\n请生成你的评论。"

                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.2,
                            max_tokens=512,
                        )
                        comment_text = completion.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"⚠️ [组合: {gender}-{age}-{edu}] 生成失败: {e}")
                        comment_text = ""

                    comments.append({
                        "gender": gender,
                        "age": age,
                        "education": edu,
                        "comment": comment_text
                    })

        # 组装结果
        result_item = {
            "id": news_id,
            "content": content,
            "label": label,
            "time": time_stamp,
            "comments": comments
        }

        # ✅ 每条写入一行 JSON（NDJSON 格式）
        fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        fout.flush()  # 立即写盘防丢数据

        print(f"[{idx+1}] ✅ 已生成 {len(comments)} 条评论。")
        elapsed = time.time() - start_time
        print(f"累计运行时间：{elapsed:.2f} 秒")

print(f"\n✅ 共处理 {len(data)} 条微博，结果已写入 {output_path}")
