import json
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Initialize BERT tokenizer and model for Chinese
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = BertModel.from_pretrained("google-bert/bert-base-chinese")

# Initialize RoBERTa tokenizer and model for English
# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")

# Define function to calculate similarity between two texts
def calculate_similarity(text1, text2):
    # Tokenize and convert to tensor
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Get sentence embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Use mean pooling of the last hidden state as sentence vector
    vector1 = outputs1.last_hidden_state.mean(dim=1)
    vector2 = outputs2.last_hidden_state.mean(dim=1)

    # Calculate cosine similarity
    similarity = F.cosine_similarity(vector1, vector2)
    return similarity.item()

# Read train.json file
with open('./train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Calculate similarity and update data for train set
for entry in data:
    td_rationale = entry.get('td_rationale', '')
    topic = entry.get('Topic', '')
    extracted_content = entry.get('extracted_content', '')
    content = entry.get('content', '')
    similarity = calculate_similarity(td_rationale, content)
    topic_similarity = calculate_similarity(topic, content)
    content_similarity = calculate_similarity(extracted_content, content)
    entry['similarity'] = similarity
    # entry['topic_similarity'] = topic_similarity
    # entry['content_similarity'] = content_similarity
    # print("topic_similarity=",topic_similarity)
    # print("content_similarity=",content_similarity)

# Write updated data back to train.json
with open('./train.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
print("train finished")

# Read test.json file
with open('./test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Calculate similarity and update data for test set
for entry in data:
    td_rationale = entry.get('td_rationale', '')
    topic = entry.get('Topic', '')
    extracted_content = entry.get('extracted_content', '')
    content = entry.get('content', '')
    similarity = calculate_similarity(td_rationale, content)
    topic_similarity = calculate_similarity(topic, content)
    content_similarity = calculate_similarity(extracted_content, content)
    entry['similarity'] = similarity
    entry['topic_similarity'] = topic_similarity
    entry['content_similarity'] = content_similarity
    # print("topic_similarity=",topic_similarity)
    # print("content_similarity=",content_similarity)

# Write updated data back to test.json
with open('./test.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
print("test finished")

# Read val.json file
with open('./val.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Calculate similarity and update data for validation set
for entry in data:
    td_rationale = entry.get('td_rationale', '')
    topic = entry.get('Topic', '')
    extracted_content = entry.get('extracted_content', '')
    content = entry.get('content', '')
    similarity = calculate_similarity(td_rationale, content)
    topic_similarity = calculate_similarity(topic, content)
    content_similarity = calculate_similarity(extracted_content, content)
    entry['similarity'] = similarity
    entry['topic_similarity'] = topic_similarity
    entry['content_similarity'] = content_similarity
    # print("topic_similarity=",topic_similarity)
    # print("content_similarity=",content_similarity)

# Write updated data back to val.json
with open('./val.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
print("val finished")