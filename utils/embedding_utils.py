from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v3"):
        self.model = SentenceTransformer(model_name)
        self.pool = self.model.start_multi_process_pool()

    def encode(self, texts):
        return self.model.encode_multi_process(texts, self.pool)

    def close(self):
        self.model.stop_multi_process_pool(self.pool)
