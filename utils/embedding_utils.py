# from sentence_transformers import SentenceTransformer

# class EmbeddingModel:
#     def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v3"):
#         self.model = SentenceTransformer(model_name, device='cuda')

#     def encode(self, texts):
#         return self.model.encode(texts, batch_size=512, convert_to_numpy=True, show_progress_bar=True)

#     def close(self):
#         pass


from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/msmarco-distilbert-base-v3"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)

        # For CPU, initialize multi-processing pool
        self.pool = None
        if self.device == 'cpu':
            self.pool = self.model.start_multi_process_pool()

    def encode(self, texts):
        if self.device == 'cuda':
            return self.model.encode(
                texts, batch_size=512, convert_to_numpy=True, show_progress_bar=True
            )
        else:
            return self.model.encode_multi_process(
                texts, self.pool, batch_size=64
            )

    def close(self):
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)
