# Created by rishijeet at 30/03/25
import torch
import torch.nn as nn
import torch.nn.functional as F


class Doc2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_docs):
        """
        Initializes the Doc2Vec model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimensionality of the embeddings.
            num_docs (int): Number of documents.
        """
        super(Doc2VecModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.doc_embeddings = nn.Embedding(num_docs, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, doc_ids, context_words):
        """
        Forward pass of the Doc2Vec model.

        Args:
            doc_ids (torch.Tensor): Tensor of document IDs.
            context_words (torch.Tensor): Tensor of context word indices.

        Returns:
            torch.Tensor: Log probabilities of the target words.
        """
        word_embeds = self.word_embeddings(context_words)
        doc_embeds = self.doc_embeddings(doc_ids)
        # Combine document and word embeddings (summation is a common choice)
        combined_embeds = word_embeds + doc_embeds
        output = self.linear(combined_embeds)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

# Train the Model
def train(model, dataloader, optimizer, epochs=10):
    """
    Trains the Doc2Vec model.

    Args:
        model (Doc2VecModel): The Doc2Vec model.
        dataloader (DataLoader): The data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            doc_ids, context_words, target_words = batch
            optimizer.zero_grad()
            log_probs = model(doc_ids, context_words)
            loss = F.nll_loss(log_probs, target_words)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    print("Finished Training")

def get_doc_embeddings(model):
    """
    Retrieves the document embeddings from the trained model.

    Args:
        model (Doc2VecModel): The trained Doc2Vec model.

    Returns:
        torch.Tensor: The document embeddings.
    """
    model.eval()
    return model.doc_embeddings.weight.detach()


def find_most_similar(target_doc_id, doc_embeddings, top_n=3):
    """
    Finds the most similar documents to a given document based on their embeddings.

    Args:
        target_doc_id (int): The ID of the target document.
        doc_embeddings (torch.Tensor): The document embeddings.
        top_n (int): Number of similar documents to return.

    Returns:
        list of int: IDs of the most similar documents.
    """
    target_embedding = doc_embeddings[target_doc_id]
    similarities = torch.cosine_similarity(target_embedding, doc_embeddings)
    _, top_indices = torch.topk(similarities, top_n + 1)  # Exclude the target document itself
    top_indices = top_indices[top_indices != target_doc_id]
    return top_indices.tolist()
