from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
if torch.cuda.is_available():
    model = model.to(torch.device('cuda'))
print("✅ Model loaded successfully")
