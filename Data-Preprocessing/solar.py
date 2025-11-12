import time  
import torch
import json
start_time = time.time()

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ====== 3. Input and output paths ======
input_path = "train.json"
output_path = "train_output.jsonl"

# ====== 4. Read JSON list ======
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ====== 5. Process each item and write in real time ======
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        content = item.get("content", "").strip()
        if not content:
            # Skip empty text
            item["td_rationale"] = ""
            item["cs_pred"] = "unknown"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{idx+1}] Empty text skipped")
            continue

        # Build prompt
        prompt = (
            f"Please extract the topic and the content of the event.The format requirement is:Topic:......Content:......:\n\n{content}\n\n"
        )
        try:
            # Tokenize and generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[{idx+1}] Model processing error: {e}")
            item["td_rationale"] = "ERROR"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            continue
        item["td_rationale"] = response

        # Write only the rationale field to output
        fout.write(json.dumps(item["td_rationale"], ensure_ascii=False) + "\n")
        print(f"[{idx+1}] Processing complete")

print(f"\n✅ All {len(data)} items processed and written to {output_path}")

# ====== 3. Input and output paths ======
input_path = "test.json"
output_path = "test_output.jsonl"

# ====== 4. Read JSON list ======
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ====== 5. Process each item and write in real time ======
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        content = item.get("content", "").strip()
        if not content:
            # Skip empty text
            item["td_rationale"] = ""
            item["cs_pred"] = "unknown"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{idx+1}] Empty text skipped")
            continue

        # Build prompt
        prompt = (
            f"Please extract the topic and the content of the event.The format requirement is:Topic:......Content:......:\n\n{content}\n\n"
        )
        try:
            # Tokenize and generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=1024)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[{idx+1}] Model processing error: {e}")
            item["td_rationale"] = "ERROR"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            continue
        item["td_rationale"] = response

        # Write only the rationale field to output
        fout.write(json.dumps(item["td_rationale"], ensure_ascii=False) + "\n")
        print(f"[{idx+1}] Processing complete")

print(f"\n✅ All {len(data)} items processed and written to {output_path}")

# ====== 3. Input and output paths ======
input_path = "val.json"
output_path = "val_output.jsonl"

# ====== 4. Read JSON list ======
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ====== 5. Process each item and write in real time ======
with open(output_path, "w", encoding="utf-8") as fout:
    for idx, item in enumerate(data):
        content = item.get("content", "").strip()
        if not content:
            # Skip empty text
            item["td_rationale"] = ""
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[{idx+1}] Empty text skipped")
            continue

        # Build prompt
        prompt = (
            f"Please extract the topic and the content of the event.The format requirement is:Topic:......Content:......:\n\n{content}\n\n"
        )

        try:
            # Tokenize and generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[{idx+1}] Model processing error: {e}")
            item["td_rationale"] = "Error"
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            continue

        item["td_rationale"] = response

        # Write only the rationale field to output
        fout.write(json.dumps(item["td_rationale"], ensure_ascii=False) + "\n")
        print(f"[{idx+1}] Processing complete")

print(f"\n✅ All {len(data)} items processed and written to {output_path}")
end_time = time.time()
print(f"Runtime: {end_time - start_time:.4f} seconds")
