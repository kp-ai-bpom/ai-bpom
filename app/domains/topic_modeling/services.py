import ast
import asyncio
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, cast

import numpy as np
import pandas as pd
from fastapi import Depends
from joblib import Parallel, delayed
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.models.ETM import ETM
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import settings
from app.core.logger import log
from app.shared.llm import init_llm

from .repositories import TopicRepository, get_topic_repository


class TopicModelingService:
    """
    Service komprehensif untuk Topic Modeling.
    Menggabungkan Preprocessing (10 Steps), LLM Augmentation, dan ETM Pipeline.
    """

    def __init__(self, repository: TopicRepository, llm: BaseChatModel):
        self.repository = repository
        self.llm = llm

        # Path dari Config (Centralized)
        self.preprocessing_path = str(settings.topic_modeling_path)
        self.utils_path = str(settings.topic_modeling_utils_path)

        self._stemmer = None  # Lazy load

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = StemmerFactory().create_stemmer()
        return self._stemmer

    # ==========================================
    # 1. ORCHESTRATOR (MAIN ENTRY POINT)
    # ==========================================
    async def process_topic_modeling(self, topic_data: dict) -> Optional[dict]:
        project_id = topic_data.get("project_id")
        keyword = topic_data.get("keyword")

        log.info(
            f"🚀 Memulai Topic Modeling Pipeline | Project: {project_id} | Keyword: {keyword}"
        )

        if not keyword:
            log.warning("Keyword is required but was None")
            return None

        # Tahap 1: Ambil Data mentah dari Repository
        result_db = await self.repository.get_tweet_by_keyword(topic_data)
        if not result_db or not result_db.get("tweets"):
            log.warning("Tidak ada tweet yang ditemukan.")
            return None

        original_tweets = result_db["tweets"]
        tweets_text = [t["full_text"] for t in original_tweets]

        # Tahap 2: Data Augmentation (LLM)
        augmented_tweets = await self._augment_data(keyword, tweets_text)

        # Tahap 3: Preprocessing Pipeline (10 Steps)
        preprocessed_df = await asyncio.to_thread(
            self._run_preprocessing_pipeline, augmented_tweets, keyword
        )

        cleaned_tweets = preprocessed_df["tweets"].tolist()

        # Mapping raw tweets
        raw_tweets_mapped = []
        for idx, cleaned_text in enumerate(cleaned_tweets):
            orig_idx = idx if idx < len(original_tweets) else idx % len(original_tweets)
            orig_tweet = original_tweets[orig_idx]
            raw_tweets_mapped.append(
                {
                    "projectId": project_id,
                    "full_text": cleaned_text,
                    "raw_text": orig_tweet.get("full_text", ""),
                    "username": orig_tweet.get("username", ""),
                    "tweet_url": orig_tweet.get("tweet_url", ""),
                    "topic": 0,
                    "probability": 0.0,
                }
            )

        # Tahap 4: ETM Model Training
        etm_result = await asyncio.to_thread(
            self._run_etm_pipeline, keyword, preprocessed_df
        )
        num_topics, model, model_output = etm_result
        topics_list = model_output.get("topics", [])

        # Tahap 5: Assign Documents & Generate Context
        documents_with_topics = self._assign_documents(raw_tweets_mapped, etm_result)

        context_data = {
            "project_id": project_id,
            "keyword": keyword,
            "num_topics": num_topics,
            "topics": topics_list,
        }
        topic_contexts = await self._generate_topic_contexts(context_data)

        # Tahap 6: Simpan ke Database secara paralel
        log.info("💾 Menyimpan hasil ke MongoDB...")
        await asyncio.gather(
            self.repository.create_documents(documents_with_topics),
            self.repository.create_topics(topic_contexts),
        )

        return {
            "project_id": project_id,
            "status": "completed",
            "keyword": keyword,
            "total_documents": len(documents_with_topics),
            "num_topics": num_topics,
            "topics": topics_list,
        }

    # ==========================================
    # 2. DATA AUGMENTATION (LLM)
    # ==========================================
    async def _augment_data(self, keyword: str, tweets: list) -> list:
        log.info("Memulai Data Augmentation dengan LLM...")
        explanation_res = cast(
            AIMessage,
            await self.llm.ainvoke(
                [
                    {
                        "role": "user",
                        "content": f"Berikan penjelasan 1 paragraf singkat tentang keyword: {keyword} di Indonesia.",
                    }
                ]
            ),
        )
        explanation = str(explanation_res.content)

        batch_size = 10
        batches = [
            tweets[i : i + batch_size] for i in range(0, len(tweets), batch_size)
        ]

        tasks = [
            self._augment_batch(batch, i + 1, keyword, explanation)
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks)

        return [tweet for batch in results for tweet in batch]

    async def _augment_batch(
        self, batch: list, batch_num: int, keyword: str, explanation: str
    ) -> list:
        prompt = f"Topic: {keyword}\nExplanation: {explanation}\nPosts: {batch}\nRephrase/Translate to formal Indonesian. Return ONLY a Python list of strings."
        try:
            res = cast(
                AIMessage, await self.llm.ainvoke([{"role": "user", "content": prompt}])
            )
            return ast.literal_eval(str(res.content).strip())
        except Exception as e:
            log.error(f"Batch {batch_num} failed augmentation: {e}. Using original.")
            return batch

    # ==========================================
    # 3. PREPROCESSING PIPELINE HELPER METHODS
    # ==========================================
    def _replace_emoticons(self, tweets: list) -> list:
        emoticon_map = {
            r":\)|:-\)|=\)": "emot-senyum",
            r":\(|:-\(|=\(": "emot-sedih",
            r":D|:-D|=D": "emot-tertawa",
            r";\)|;-\)": "emot-mengedip",
            r":P|:-P|=P": "emot-julur",
            r":O|:-O|=O": "emot-terkejut",
            r":\/|:-\\": "emot-bingung",
            r"<3": "emot-hati",
            r":\*|:-\*": "emot-ciuman",
        }
        result = []
        for tweet in tweets:
            for pattern, replacement in emoticon_map.items():
                tweet = re.sub(pattern, replacement, tweet)
            result.append(tweet)
        return result

    def _delete_extra_letters(self, tokenized_tweets: list) -> list:
        seq_pattern = r"([A-Za-z])\1{2,}"
        return [
            [re.sub(seq_pattern, r"\1", token) for token in tokens]
            for tokens in tokenized_tweets
        ]

    def _normalization(self, tokenized_tweets: list) -> list:
        kbba_path = os.path.join(self.utils_path, "kbba.txt")
        try:
            with open(kbba_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                kbba_data = [line.strip().split("\t") for line in lines if "\t" in line]
                kontraksi_dict = dict(kbba_data)

            return [
                [kontraksi_dict.get(word, word) for word in tokens]
                for tokens in tokenized_tweets
            ]
        except FileNotFoundError:
            log.warning(f"KBBA file not found at {kbba_path}. Skipping normalization.")
            return tokenized_tweets

    def _curating_stopword(self, tokenized_tweets: list) -> set:
        corpus = [" ".join(tokens) for tokens in tokenized_tweets]

        # 1. TF-IDF
        tr_idf_model = TfidfVectorizer()
        tf_idf_vector = tr_idf_model.fit_transform(corpus)
        df_tf_idf = pd.DataFrame(
            tf_idf_vector.toarray(),  # type: ignore[misc]
            columns=tr_idf_model.get_feature_names_out(),
        )
        high_tfidf_words = df_tf_idf.columns[(df_tf_idf > 0.7).any()].tolist()

        # 2. Rare words
        word_freq = Counter(word for tokens in tokenized_tweets for word in tokens)
        num_tweets = len(tokenized_tweets)

        if num_tweets >= 10000:
            rare_words = [word for word, freq in word_freq.items() if freq <= 10]
        elif num_tweets >= 100:
            rare_words = [word for word, freq in word_freq.items() if freq < 2]
        else:
            rare_words = []

        # 3. PRON & Manual
        PRON = [
            "aku",
            "saya",
            "gue",
            "gw",
            "kamu",
            "kau",
            "engkau",
            "dia",
            "ia",
            "kita",
            "kami",
            "mereka",
            "anda",
            "lo",
            "lu",
            "kalian",
        ]
        manual_stopwords = [
            "aduh",
            "sangat",
            "amp",
            "the",
            "link",
            "yang",
            "iya",
            "ada",
            "tin",
            "sangat",
            "tidak",
            "jadi",
            "mungkin",
            "apa",
            "orang",
            "wah",
        ]

        return set(high_tfidf_words + rare_words + PRON + manual_stopwords)

    def _run_preprocessing_pipeline(self, tweets: list, keyword: str) -> pd.DataFrame:
        log.info("Menjalankan 10-Step Preprocessing Pipeline...")
        initial_count = len(tweets)

        # Step 1: Remove URL
        data = [
            re.sub(
                r"(?:https?://|www\.)(?:[^\s./]+\.)+[^\s./]+(?:/\S*)?", "", t
            ).strip()
            for t in tweets
        ]

        # Step 2: Replace Emoticons
        data = self._replace_emoticons(data)

        # Step 3: Remove Twitter symbols
        data = [re.sub(r"#\w+|@\w+|\bRT\b", " ", t) for t in data]

        # Step 4: Remove symbols and punctuation
        data = [re.sub(r"[^a-zA-Z\s]", " ", t).strip() for t in data]
        data = [re.sub(r"\s+", " ", t).strip() for t in data]

        # Step 5 & 6: Tokenizing & Case folding
        data = [[token.lower() for token in t.split()] for t in data]

        # Step 7: Delete extra letters
        data = self._delete_extra_letters(data)

        # Step 8: Normalization
        data = self._normalization(data)

        # Step 9: Stemming (Parallel)
        with ThreadPoolExecutor(max_workers=4) as executor:
            data = list(
                executor.map(
                    lambda tokens: [self.stemmer.stem(tok) for tok in tokens], data
                )
            )

        # Step 10: Stopword Removal dengan Curating TF-IDF
        custom_stopwords = self._curating_stopword(data)
        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

        final_data = []
        for tokens in data:
            cleaned = stopword_remover.remove(" ".join(tokens))
            final_data.append(
                [
                    t
                    for t in cleaned.split()
                    if len(t) > 2 and t.lower() not in custom_stopwords
                ]
            )

        # Hapus tweet kosong
        final_data = [t for t in final_data if t]

        df = pd.DataFrame({"tweets": final_data})
        df["label"] = np.where(
            df.index < int(0.85 * len(df)),
            "train",
            np.where(df.index < int(0.90 * len(df)), "val", "test"),
        )
        df["tweets"] = df["tweets"].apply(lambda x: " ".join(x))

        # Save OCTIS Format
        folder_name = f"octis_data-{keyword.replace(' ', '_')}"
        out_path = os.path.join(self.preprocessing_path, "vocabs", folder_name)
        os.makedirs(out_path, exist_ok=True)

        vocab = set(word for text in df["tweets"] for word in text.split())
        with open(os.path.join(out_path, "vocabulary.txt"), "w") as f:
            for w in sorted(vocab):
                f.write(w + "\n")

        df[["tweets", "label"]].to_csv(
            os.path.join(out_path, "corpus.tsv"), sep="\t", index=False, header=False
        )

        log.info(
            f"Preprocessing selesai: {len(df)}/{initial_count} tweet berhasil diproses."
        )
        return df

    # ==========================================
    # 4. ETM MODELING
    # ==========================================
    def _run_etm_pipeline(self, keyword: str, df: pd.DataFrame) -> tuple:
        log.info("Mempersiapkan ETM Model...")
        folder_name = f"octis_data-{keyword.replace(' ', '_')}"
        dataset_path = os.path.join(self.preprocessing_path, "vocabs", folder_name)

        dataset = Dataset()
        dataset.load_custom_dataset_from_folder(dataset_path)

        def train_model(num_topics):
            emb_path = os.path.join(
                self.preprocessing_path, "wiki", "idwiki_word2vec_100_new_lower.txt"
            )
            model = ETM(
                num_topics=num_topics,
                num_epochs=100,
                batch_size=256,
                dropout=0.3,
                activation="tanh",
                embeddings_path=emb_path,
                embeddings_type="word2vec",
                t_hidden_size=512,
                wdecay=1e-5,
                lr=0.001,
                optimizer="SGD",
            )
            out = model.train_model(dataset)
            score = Coherence(texts=dataset.get_corpus(), topk=10, measure="c_v").score(
                out
            )
            return num_topics, model, out, score

        # Pencarian jumlah topik optimal (Kembali ke range 1-7 sesuai kode awal)
        topics_range = range(1, 7)
        results = cast(
            list,
            Parallel(n_jobs=-1, backend="threading")(
                delayed(train_model)(n) for n in topics_range
            ),
        )
        if not results:
            raise RuntimeError("ETM training failed - no results returned")

        best_result = max(results, key=lambda item: item[3])
        log.info(
            f"Topik optimal terpilih: {best_result[0]} (Coherence C_V: {best_result[3]:.4f})"
        )

        return best_result[0], best_result[1], best_result[2]

    # ==========================================
    # 5. ASSIGNMENT & CONTEXT
    # ==========================================
    def _assign_documents(self, raw_tweets_mapped: list, etm_model: tuple) -> list:
        probs = etm_model[2]["topic-document-matrix"]
        num_docs = probs.shape[1]

        for i in range(min(num_docs, len(raw_tweets_mapped))):
            column = probs[:, i]
            topic_idx = int(np.argmax(column))
            raw_tweets_mapped[i]["topic"] = topic_idx
            raw_tweets_mapped[i]["probability"] = float(column[topic_idx])

        return raw_tweets_mapped

    async def _generate_topic_contexts(self, context_data: dict) -> list:
        prompt = f'Topik ini membahas tentang keyword: {context_data["keyword"]}. Hasil topic modeling dengan {context_data["num_topics"]} topik: {context_data["topics"]}\nBuatkan dengan format JSON dengan 1 topik untuk 1 kalimat utama. Format:\n[\n  {{\n    "kata_kunci": "..."\n    "kalimat": "Topik ini tentang ..."\n  }}\n]\nONLY answer in JSON FORMAT without opening words.'

        res = cast(
            AIMessage, await self.llm.ainvoke([{"role": "user", "content": prompt}])
        )
        json_text = re.search(
            r"\[(?:\s*{[^{}]*}\s*,?)*\s*\]",
            str(res.content).replace("\n", ""),
            re.DOTALL,
        )

        results = []
        if json_text:
            parsed = json.loads(json_text.group())
            for idx, item in enumerate(parsed):
                results.append(
                    {
                        "projectId": context_data["project_id"],
                        "topicId": idx,
                        "keyword": context_data["keyword"],
                        "words": [
                            w.strip() for w in item.get("kata_kunci", "").split(",")
                        ],
                        "context": item.get("kalimat", ""),
                    }
                )
        return results


# ==========================================
# DEPENDENCY FACTORY
# ==========================================
def get_topic_modeling_service(
    repo: TopicRepository = Depends(get_topic_repository),
    llm: BaseChatModel = Depends(init_llm),
) -> TopicModelingService:
    return TopicModelingService(repo, llm)
