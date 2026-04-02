import asyncio
import os
import re
import string
from typing import Optional

import pandas as pd
from fastapi import Depends
from keras.preprocessing.sequence import pad_sequences
from mpstemmer import MPStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from app.core.config import settings
from app.core.logger import log
from app.domains.topic_modeling.models import DocumentsModel

from .engine import SentimentModelManager, get_sentiment_models
from .repositories import SentimentRepository, get_sentiment_repository


class SentimentService:
    def __init__(self, repository: SentimentRepository, models: SentimentModelManager):
        self.repository = repository

        self.cnn_model = models.cnn_model
        self.cnn_tokenizer = models.cnn_tokenizer
        self.cnn_lstm_model = models.cnn_lstm_model
        self.cnn_lstm_tokenizer = models.cnn_lstm_tokenizer

        # Path dari Config
        self.utils_path = str(settings.sentiment_utils_path)

        self.max_len = 50
        self.stemmer = MPStemmer()

        self.normalization_dict = self._load_normalization_dict()
        self.stop_words = self._load_stopwords()

    def _load_normalization_dict(self) -> dict:
        try:
            dict_path = os.path.join(self.utils_path, "kamus.csv")
            if os.path.exists(dict_path):
                df = pd.read_csv(dict_path, header=None)
                return {row[0]: row[1] for _, row in df.iterrows()}
        except Exception as e:
            log.error(f"Gagal memuat kamus normalisasi: {e}")
        return {}

    def _load_stopwords(self) -> set:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        stops = stopwords.words("indonesian")
        negation_words = {
            "tidak",
            "baik",
            "jelek",
            "jangan",
            "belum",
            "bukan",
            "enggak",
            "engga",
            "bener",
            "benar",
        }
        stops = [w for w in stops if w not in negation_words]

        custom_stopwords = [
            "yg",
            "dg",
            "rt",
            "dgn",
            "ny",
            "d",
            "klo",
            "kalo",
            "amp",
            "biar",
            "bikin",
            "bilang",
            "gak",
            "ga",
            "krn",
            "nya",
            "nih",
            "sih",
            "si",
            "tau",
            "tdk",
            "tuh",
            "utk",
            "ya",
            "jd",
            "jgn",
            "sdh",
            "aja",
            "n",
            "t",
            "nyg",
            "hehe",
            "pen",
            "u",
            "nan",
            "loh",
            "&amp",
            "yah",
        ]
        stops.extend(custom_stopwords)
        return set(stops)

    def _preprocess_single_text(self, text: str) -> str:
        # 1. Case Folding & 2. Special Chars
        text = text.lower().replace("\\t", " ").replace("\\n", " ").replace("\\", "")
        text = text.encode("ascii", "ignore").decode("ascii")
        text = " ".join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())

        # 3. Numbers, 4. Punctuation, 5. Whitespace
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation)).strip()
        text = re.sub(r"\s+", " ", text)

        # 6. Single Chars & 7. Repeated Chars
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        text = re.sub(r"(.)\1+", r"\1", text)

        # 8-12. Tokenize, Normalize, Stem, Stopword, Negation
        try:
            tokens = word_tokenize(text)
        except LookupError:
            import nltk

            nltk.download("punkt")
            tokens = word_tokenize(text)

        tokens = [self.normalization_dict.get(t, t) for t in tokens]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stop_words]

        negations = {"tidak", "jangan", "belum", "bukan", "enggak"}
        result = []
        skip = False
        for i, word in enumerate(tokens):
            if skip:
                skip = False
                continue
            if word in negations and i + 1 < len(tokens):
                result.append(f"{word}_{tokens[i + 1]}")
                skip = True
            else:
                result.append(word)

        return " ".join(result)

    def _run_inference(self, texts: list, model, tokenizer) -> list:
        if not model or not tokenizer:
            return [("Netral", 0.0)] * len(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences, maxlen=self.max_len, truncating="post", padding="post"
        )
        predictions = model.predict(padded, verbose=0)

        results = []
        for pred in predictions:
            prob_pos = float(pred[0])
            label = "Positif" if prob_pos > 0.5 else "Negatif"
            conf = prob_pos if label == "Positif" else (1 - prob_pos)
            results.append((label, conf))
        return results

    async def process_sentiment(self, project_id: str) -> Optional[dict]:
        log.info(f"Mulai klasifikasi sentimen untuk project: {project_id}")

        documents = await DocumentsModel.find(
            DocumentsModel.projectId == project_id
        ).to_list()
        if not documents:
            return None

        raw_texts = [doc.raw_text for doc in documents]
        log.info("Menjalankan 12-Step Preprocessing...")
        preprocessed_texts = await asyncio.to_thread(
            lambda texts: [self._preprocess_single_text(t) for t in texts], raw_texts
        )

        log.info("Menjalankan Inferensi Model CNN & CNN-LSTM...")
        cnn_preds = await asyncio.to_thread(
            self._run_inference, preprocessed_texts, self.cnn_model, self.cnn_tokenizer
        )
        lstm_preds = await asyncio.to_thread(
            self._run_inference,
            preprocessed_texts,
            self.cnn_lstm_model,
            self.cnn_lstm_tokenizer,
        )

        result_docs = []
        for i, doc in enumerate(documents):
            result_docs.append(
                {
                    "projectId": project_id,
                    "raw_text": doc.raw_text,
                    "preprocessed_text": preprocessed_texts[i],
                    "topic": doc.topic,
                    "sentiment_cnn": cnn_preds[i][0],
                    "sentiment_cnn_probability": cnn_preds[i][1],
                    "sentiment_cnn_lstm": lstm_preds[i][0],
                    "sentiment_cnn_lstm_probability": lstm_preds[i][1],
                }
            )

        await self.repository.save_sentiments(result_docs)

        def calc_percentages(data_list, key):
            total = len(data_list)
            if total == 0:
                return {"positive": 0, "negative": 0}
            pos = sum(1 for d in data_list if d[key] == "Positif")
            return {
                "positive": round((pos / total) * 100, 2),
                "negative": round(((total - pos) / total) * 100, 2),
            }

        def calc_by_topic(data_list, key):
            topics = {}
            for d in data_list:
                t = str(d["topic"]) if d["topic"] is not None else "unknown"
                if t not in topics:
                    topics[t] = {"total": 0, "pos": 0}
                topics[t]["total"] += 1
                if d[key] == "Positif":
                    topics[t]["pos"] += 1

            return {
                t: {
                    "total": v["total"],
                    "positive": round((v["pos"] / v["total"]) * 100, 2),
                    "negative": round(((v["total"] - v["pos"]) / v["total"]) * 100, 2),
                }
                for t, v in topics.items()
            }

        return {
            "project_id": project_id,
            "total": len(result_docs),
            "documents": result_docs,
            "sentiment_percentage_cnn": calc_percentages(result_docs, "sentiment_cnn"),
            "sentiment_percentage_cnn_lstm": calc_percentages(
                result_docs, "sentiment_cnn_lstm"
            ),
            "sentiment_by_topic_cnn": calc_by_topic(result_docs, "sentiment_cnn"),
            "sentiment_by_topic_cnn_lstm": calc_by_topic(
                result_docs, "sentiment_cnn_lstm"
            ),
        }


# ==========================================
# 3. DEPENDENCY INJECTION FACTORY
# ==========================================
def get_sentiment_service(
    repo: SentimentRepository = Depends(get_sentiment_repository),
    models: SentimentModelManager = Depends(get_sentiment_models),
) -> SentimentService:
    return SentimentService(repo, models)
