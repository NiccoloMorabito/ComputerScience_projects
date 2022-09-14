NON_TARGET_CLASS = 0
CLASS_TO_SENTIMENT = {1: 'conflict', 2: 'neutral', 3: 'positive', 4: 'negative'}
SENTIMENT_TO_CLASS = {'conflict': 1, 'neutral': 2, 'positive': 3, 'negative': 4}

ALL_CATEGORIES = ['food', 'anecdotes/miscellaneous', 'price', 'service', 'ambience']

# validation_metrics
VALID_F1 = "valid_f1"
VALID_F1_MACRO = "valid_f1_macro"

BATCH_INPUT_INDICES = "input_embedding_indices"
BATCH_INPUT_LENGTHS = "input_lengths"
BATCH_TARGET_BOOLEANS = "target_booleans"
BATCH_TARGET_CLASSES = "target_classes"
BATCH_SAMPLE = "samples"
BATCH_BERT_ENCODINGS = "bert_encodings"
BATCH_BERT_TARGET_BOOLEANS = "bert_labels_booleans"
BATCH_BERT_TARGET_CLASSES = "bert_labels_classes"
BATCH_CATEGORY_BOOLEANS = "category_booleans"
BATCH_CATEGORY_CLASSES = "category_classes"
