from datasets import load_dataset
from transformers import AutoTokenizer

from config import Config

tokenizer = AutoTokenizer.from_pretrained(
    Config.MODEL_NAME
)