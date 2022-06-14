from typing import Any

from catboost import CatBoostClassifier


class CatBoostWrapper:

    def __init__(
            self,
            **kwargs: Any
    ):
        self.model: CatBoostClassifier = CatBoostClassifier(
            **kwargs,
            text_processing={
                "tokenizers": [{
                    "tokenizer_id": "Space",
                    "separator_type": "ByDelimiter",
                    "delimiter": " "
                }],

                "dictionaries": [{
                    "dictionary_id": "BiGram",
                    "token_level_type": "Letter",
                    "max_dictionary_size": "150000",
                    "occurrence_lower_bound": "1",
                    "gram_order": "2"
                }, {
                    "dictionary_id": "Trigram",
                    "max_dictionary_size": "150000",
                    "token_level_type": "Letter",
                    "occurrence_lower_bound": "1",
                    "gram_order": "3"
                }, {
                    "dictionary_id": "Fourgram",
                    "max_dictionary_size": "150000",
                    "token_level_type": "Letter",
                    "occurrence_lower_bound": "1",
                    "gram_order": "4"
                }, {
                    "dictionary_id": "Fivegram",
                    "max_dictionary_size": "150000",
                    "token_level_type": "Letter",
                    "occurrence_lower_bound": "1",
                    "gram_order": "5"
                }, {
                    "dictionary_id": "Sixgram",
                    "max_dictionary_size": "150000",
                    "token_level_type": "Letter",
                    "occurrence_lower_bound": "1",
                    "gram_order": "6"
                }
                ],

                "feature_processing": {
                    "default": [
                        {
                            "dictionaries_names": ["BiGram", "Trigram",
                                                   "Fourgram", "Fivegram",
                                                   "Sixgram"],
                            "feature_calcers": ["BoW"],
                            "tokenizers_names": ["Space"]
                        },
                        {
                            "dictionaries_names": ["BiGram", "Trigram",
                                                   "Fourgram", "Fivegram",
                                                   "Sixgram"],
                            "feature_calcers": ["NaiveBayes"],
                            "tokenizers_names": ["Space"]
                        }, {
                            "dictionaries_names": ["BiGram", "Trigram",
                                                   "Fourgram", "Fivegram",
                                                   "Sixgram"],
                            "feature_calcers": ["BM25"],
                            "tokenizers_names": ["Space"]
                        },
                    ],
                }
            }
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)