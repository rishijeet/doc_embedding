# Created by rishijeet at 30/03/25
from torch.utils.data import Dataset
from data.DataProcessor import text_to_sequence, preprocess_text

class Doc2VecDataset(Dataset):
    def __init__(self, texts, vocab, window_size=5):
        """
        Initializes the Doc2Vec dataset.

        Args:
            texts (list of str): List of documents (strings).
            vocab (dict): Vocabulary mapping words to indices.
            window_size (int): Size of the context window.
        """
        self.texts = [text_to_sequence(preprocess_text(text), vocab) for text in texts]
        self.vocab = vocab
        self.window_size = window_size
        self.doc_ids = []
        self.context_words = []
        self.target_words = []

        for doc_id, text in enumerate(self.texts):
            for i, target_word_idx in enumerate(text):
                # Ensure the window doesn't go out of bounds
                start = max(0, i - window_size)
                end = min(len(text), i + window_size + 1)
                context_word_idxs = []
                for j in range(start, end):
                    if j != i:
                        context_word_idxs.append(text[j])
                self.doc_ids.extend([doc_id] * len(context_word_idxs))
                self.context_words.extend(context_word_idxs)
                self.target_words.extend([target_word_idx] * len(context_word_idxs))

    def __len__(self):
        return len(self.context_words)

    def __getitem__(self, idx):
        return (
            self.doc_ids[idx],
            self.context_words[idx],
            self.target_words[idx],
        )
