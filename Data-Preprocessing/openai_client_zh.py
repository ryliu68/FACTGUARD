import time
import json
from openai import OpenAI


# ========== 1. Initialize the client ==========
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="xxxx"
)

model_name ="deepseek-v3.1"
# model_name = "qwen3-235b-a22b-instruct-2507"

start_time = time.time()

# ========== 2. Input and output paths ==========
input_path = "train.json"
output_path = "train_event.json"

# ========== 3. Load input data ==========
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# ========== 4. Write output ==========
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        content = item.get("content", "").strip()

        if not content:
            item["td_rationale"] = ""
            results.append(item)
            print(f"[{idx+1}] Empty text skipped")
            continue

        # ========= Chinese task prompt (kept in Chinese) =========
        prompt = f"""
你是一个中文新闻信息抽取助手，请阅读下面的新闻文本，从中抽取两个要素：
1. **Topic（主题）**：用一句简洁的话概括新闻的核心事件或主要主题。
2. **Content（内容摘要）**：用简洁流畅的中文总结新闻的主要事实、背景或观点。

请严格按照以下格式输出：
Topic：...
Content：...

---
示例：
输入文本：
【媒体融合链来了中国搜索赋能版权保护】媒体融合链，版权价值链。由新华社中国搜索研发的媒体融合链区块链版权平台，近日正式上线。作为国家版权局版权保护新技术研究推广站点授牌单位，中国搜索此举旨在打造融合、协同、创新的版权新生态，助力提升媒体版权价值。

输出：
Topic：新华社中国搜索研发的区块链版权平台上线，赋能版权保护。
Content：由新华社中国搜索研发的媒体融合链区块链版权平台近日正式上线，作为国家版权局版权保护新技术研究推广站点授牌单位，中国搜索通过此举，旨在打造融合、协同、创新的版权新生态，助力提升媒体版权价值。

---
现在请处理以下新闻文本：
{content}
"""

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个中文新闻主题与摘要提取助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )

            response = completion.choices[0].message.content.strip()
            # print(f"[{idx+1}] Output:\n{response}\n")

            # Only replace td_rationale, keep other fields unchanged
            item["td_rationale"] = response

        except Exception as e:
            print(f"[{idx+1}] Model processing error: {e}")
            item["td_rationale"] = "ERROR"

        results.append(item)

        # Write results in real-time to avoid data loss in case of interruption
        fout.seek(0)
        json.dump(results, fout, ensure_ascii=False, indent=2)

end_time = time.time()
print(f"\n✅ Processed {len(results)} news items, results saved to {output_path}")
print(f"Execution time: {end_time - start_time:.2f} seconds")
