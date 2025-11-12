
### Code introduction
This module demonstrates how data processing is conducted:

**FACTGUARD:** `deepseek.py` and `solar.py` use locally deployed models from Hugging Face to process datasets. They can successfully handle the vast majority of news articles. Afterwards, `similar.py`, `zh_shannon.py`, and `en_shannon.py`are used for evaluation; items that do not meet the criteria are filtered out. Then, `DeepSeek_v3.1.py` (with few-shot added) is used to process the filtered data, followed by another evaluation. (Alternatively, `openai_client_zh.py和openai_client_en.py` can be used directly from the start—note that the focus is to satisfy the cosine similarity filtering conditions as much as possible.)

**GenFEND:** `GenFEND_en.py、GenFEND_en.py`contains my reproduction of the GenFEND authors’ data processing code.
