import time
import json
from openai import OpenAI


client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="xxxx"
)

model_name="deepseek-v3.1""
# model_name = "qwen3-235b-a22b-instruct-2507"

start_time = time.time()

input_path = "val.json"
output_path = "val_event.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        content = item.get("content", "").strip()

        if not content:
            item["td_rationale"] = ""
            results.append(item)
            print(f"[{idx+1}] Empty text skipped")
            continue

        # Few-shot Prompt Template
        prompt = f"""
        You are an information extraction assistant. Your task is to read a given passage and extract two elements:
        1. **Topic** – A short phrase summarizing the main event or focus.
        2. **Content** – A concise description summarizing what happened.

        Please follow the format strictly:
        Topic: ...
        Content: ...

        ---

        ### Example

        Input Text:
        No matter what people thought of "The Bachelorette" Season 13 finale on Monday, August 7, they must admit Rachel Lindsay's proposal dress was elegant. However, she didn't get to keep it. What viewers didn't know is how much her dress weighed or how much her engagement ring cost. 30-pound dress Viewers probably heard Rachel exclaim that the dress was heavy when Bryan Abasolo picked her up after he proposed to her. He quickly put her down before they made their way downhill from the castle. You may like Meghan Markle has no objection if Prince Harry shoots pheasant on Boxing Day Rachel did not exaggerate. The designer admitted the dress did weigh 30 pounds because of the beads. The gown Rachel wore appears in Randi Rahm.

        Output:
        Topic: The Bachelorette Season 13 finale and Rachel Lindsay's proposal dress
        Content: The finale of Season 13 of The Bachelorette aired on August 7. Rachel Lindsay's proposal dress was elegant, but it weighed 30 pounds due to the beads. The designer, Randi Rahm, confirmed the dress's weight. The engagement ring's cost was not mentioned. After the proposal, Bryan Abasolo picked Rachel up, and she mentioned the dress's weight. He quickly put her down before they walked downhill from the castle. The dress was from Randi Rahm's collection.

        ---

        Now, extract the topic and content for the following text:

        Input Text:
        {content}

        Output:
        Topic:
        Content:
        """

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts structured information from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512,
            )

            response = completion.choices[0].message.content.strip()
            print(f"[{idx+1}] Output:\n{response}\n")

            # Replace only td_rationale, keep other fields unchanged
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
