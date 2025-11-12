import time
import json
from openai import OpenAI

# ========== 1. Initialize client ==========
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="xxxx"
)

model_name = "qwen3-235b-a22b-instruct-2507"  # Elapsed time: 71.69s

start_time = time.time()

# ========== 2. Attribute definitions ==========
genders = ["male", "female"]
ages = [
    "You are under 17 years old.",
    "You are 18 to 29 years old.",
    "You are 30 to 49 years old.",
    "You are 50 to 64 years old.",
    "You are over 65 years old.",
]
educations = [
    "Educationally, you have a high school diploma or less.",
    "Educationally, you haven't graduated from college.",
    "Educationally, you are a college grad.",
]

# ========== 3. Input / Output paths ==========
input_path = "./train.json"          # 
output_path = "./train_comments.json"  # 

# ========== 4. Read input data ==========
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========== 5. Generate comments and write line by line ==========
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        title = item.get("source_id", "")
        content = item.get("content", "").strip()
        label = item.get("label", "")
        if not content:
            print(f"[{idx+1}] Empty text, skipped.")
            continue

        comments = []

        # Iterate all 30 identity combinations
        for gender in genders:
            for age in ages:
                for edu in educations:
                    system_prompt = (
                        f"Suppose you are a {gender} Twitter user. "
                        f"{age} {edu} "
                        "You will be provided with an article. "
                        "You should write one comment about the article. "
                        "Note that your comment needs to match your identity, "
                        "and should be brief and natural, like normal Twitter users."
                    )
                    user_prompt = f"news: {content}"

                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.4,
                            max_tokens=512,
                        )
                        comment_text = completion.choices[0].message.content.strip()
                    except Exception as e:
                        print(f"⚠️ [Combo: {gender}-{age}-{edu}] Generation failed: {e}")
                        comment_text = ""

                    comments.append({
                        "gender": gender,
                        "age": age,
                        "education": edu,
                        "comment": comment_text
                    })

        # Combine the result for this article
        result_item = {
            "title": title,
            "content": content,
            "label": label,
            "comments": comments
        }

        fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        fout.flush()

        print(f"[{idx+1}] Generated {len(comments)} comments.")
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f}s")

print(f"\n✅ All {len(data)} news articles processed and saved to {output_path}.")
