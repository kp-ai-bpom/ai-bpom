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

from .engine import EmotionModelManager, get_emotion_models
from .repositories import EmotionRepository, get_emotion_repository


class EmotionService:
    def __init__(self, repository: EmotionRepository, models: EmotionModelManager):
        self.repository = repository

        self.cnn_model = models.cnn_model
        self.cnn_tokenizer = models.cnn_tokenizer
        self.bilstm_model = models.bilstm_model
        self.bilstm_tokenizer = models.bilstm_tokenizer
        self.labels = models.emotion_labels

        # Path dari Config
        self.utils_path = str(settings.emotion_utils_path)

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

        stopword_path = os.path.join(self.utils_path, "stopwords.txt")
        if os.path.exists(stopword_path):
            with open(stopword_path, "r", encoding="utf-8") as f:
                stops.extend(f.read().split())

        return set(stops)

    def _preprocess_single_text(self, text: str) -> str:
        text = text.lower().replace("\\t", " ").replace("\\n", " ").replace("\\", "")
        text = text.encode("ascii", "ignore").decode("ascii")
        text = " ".join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())

        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation)).strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b[a-zA-Z]\b", "", text)
        text = re.sub(r"(.)\1+", r"\1", text)

        try:
            tokens = word_tokenize(text)
        except LookupError:
            import nltk

            nltk.download("punkt")
            tokens = word_tokenize(text)

        tokens = [self.normalization_dict.get(t, t) for t in tokens]
        tokens = [self.stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stop_words]
        # NEGATION TIDAK DIAPLIKASIKAN (Sesuai Worker Anda)
        return " ".join(tokens)

    def _run_inference(self, texts: list, model, tokenizer) -> list:
        if not model or not tokenizer:
            # Fallback format: (Label, Dictionary of Probabilities)
            fallback_probs = {label: 0.0 for label in self.labels}
            fallback_probs["Neutral"] = 1.0
            return [("Neutral", fallback_probs)] * len(texts)

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences, maxlen=self.max_len, truncating="post", padding="post"
        )
        predictions = model.predict(padded, verbose=0)

        results = []
        for pred in predictions:
            pred_probs = {
                self.labels[i]: float(pred[i]) for i in range(len(self.labels))
            }
            top_label = max(pred_probs, key=pred_probs.__getitem__)
            results.append((top_label, pred_probs))

        return results

    async def process_emotion(self, project_id: str) -> Optional[dict]:
        log.info(f"Mulai klasifikasi Emosi untuk project: {project_id}")

        # 1. Ambil dokumen dari koleksi Topic Modeling
        documents = await DocumentsModel.find(
            DocumentsModel.projectId == project_id
        ).to_list()
        if not documents:
            return None

        # 2. Preprocessing
        raw_texts = [doc.raw_text for doc in documents]
        preprocessed_texts = await asyncio.to_thread(
            lambda texts: [self._preprocess_single_text(t) for t in texts], raw_texts
        )

        # 3. Keras Inference
        log.info("Menjalankan Inferensi Emotion CNN & BiLSTM...")
        cnn_preds = await asyncio.to_thread(
            self._run_inference, preprocessed_texts, self.cnn_model, self.cnn_tokenizer
        )
        bilstm_preds = await asyncio.to_thread(
            self._run_inference,
            preprocessed_texts,
            self.bilstm_model,
            self.bilstm_tokenizer,
        )

        # 4. Format Dokumen
        result_docs = []
        for i, doc in enumerate(documents):
            result_docs.append(
                {
                    "projectId": project_id,
                    "raw_text": doc.raw_text,
                    "preprocessed_text": preprocessed_texts[i],
                    "topic": doc.topic,
                    "emotion_cnn": cnn_preds[i][0],
                    "emotion_cnn_probability": cnn_preds[i][1],
                    "emotion_bilstm": bilstm_preds[i][0],
                    "emotion_bilstm_probability": bilstm_preds[i][1],
                }
            )

        # 5. Simpan ke MongoDB
        await self.repository.save_emotions(result_docs)

        # 6. Agregasi Statistik
        def calc_percentages(data_list, key):
            total = len(data_list)
            if total == 0:
                return {label: 0.0 for label in self.labels}
            counts = {label: 0 for label in self.labels}
            for d in data_list:
                counts[d[key]] += 1
            return {
                label: round((counts[label] / total) * 100, 2) for label in self.labels
            }

        def calc_by_topic(data_list, key):
            topics = {}
            for d in data_list:
                t = str(d["topic"]) if d["topic"] is not None else "unknown"
                if t not in topics:
                    topics[t] = {label: 0 for label in self.labels}
                    topics[t]["total"] = 0
                topics[t]["total"] += 1
                topics[t][d[key]] += 1

            res = {}
            for t, counts in topics.items():
                res[t] = {
                    label: round((counts[label] / counts["total"]) * 100, 2)
                    for label in self.labels
                }
            return res

        return {
            "project_id": project_id,
            "total": len(result_docs),
            "documents": result_docs,
            "emotion_percentage_cnn": calc_percentages(result_docs, "emotion_cnn"),
            "emotion_percentage_bilstm": calc_percentages(
                result_docs, "emotion_bilstm"
            ),
            "emotion_by_topic_cnn": calc_by_topic(result_docs, "emotion_cnn"),
            "emotion_by_topic_bilstm": calc_by_topic(result_docs, "emotion_bilstm"),
        }


def get_emotion_service(
    repo: EmotionRepository = Depends(get_emotion_repository),
    models: EmotionModelManager = Depends(get_emotion_models),
) -> EmotionService:
    return EmotionService(repo, models)
