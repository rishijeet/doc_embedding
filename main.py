import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.DataProcessor import build_vocab
from data.Doc2VecDataset import Doc2VecDataset
from model.Doc2VecModel import Doc2VecModel, train, get_doc_embeddings, find_most_similar

# Example usage:
texts = [
    "This is the first document.",
    "This is the second document.",
    "A third document is here.",
    "The first document again."
]
vocab = build_vocab(texts)
dataset = Doc2VecDataset(texts, vocab, window_size=2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

vocab_size = len(vocab) + 1 # +1 for unknown words. Important for the nn.Embedding
embed_size = 100 # added the sizing
num_docs = len(texts) # the input added
model = Doc2VecModel(vocab_size, embed_size, num_docs)
optimizer = optim.Adam(model.parameters(), lr=0.001)


train(model, dataloader, optimizer, epochs=10)

# Get the embedding
doc_embeddings = get_doc_embeddings(model)
print("Document Embeddings:", doc_embeddings)

# Example: Find the 3 most similar documents to the first document (doc_id=0)
similar_docs = find_most_similar(0, doc_embeddings)
print(f"Most similar documents to document 0: {similar_docs}")
