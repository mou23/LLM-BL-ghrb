from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v3"):
        self.model = SentenceTransformer(model_name, device='cuda')

    def encode(self, texts):
        return self.model.encode(texts, batch_size=512, convert_to_numpy=True, show_progress_bar=True)

    def close(self):
        pass
