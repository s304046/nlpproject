from typing import List, Optional, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from torch import nn, Tensor
import re

class DiscriminatorModel(nn.Module):
    
    def __init__(
        self,
        model_name_or_path: str,
        pretrained_path: str = None,
        max_seq_length: int = 64,
        truncation: str = "longest_first",
        padding: str = "max_length",
        extra_tokens: List[str] = None  # extra words
        ):
        super(DiscriminatorModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        self.padding = padding

        if pretrained_path is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_path}tokenizer/")

        if extra_tokens:  # ADD EXTRA TOKENS IF PROVIDED
            new_tokens = [tok for tok in extra_tokens if tok not in self.tokenizer.get_vocab()]
            if new_tokens:
                num_added_toks = self.tokenizer.add_tokens(new_tokens)
                print(f"Added {num_added_toks} new tokens when Instantiating discriminator: {new_tokens}")

                # **Resize token embeddings to accommodate new tokens**
                self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self):
        # Setting the model in training mode
        self.model.train()

    def eval(self):
        # Setting the model in evaluation mode
        self.model.eval()

    @staticmethod
    def extract_new_tokens(sentences: List[str], tokenizer):
        # Extracts words not present in the tokenizer vocabulary
        existing_vocab = tokenizer.get_vocab()
        words = set()
        for sentence in sentences:
            clean_words = re.findall(r"\b\w+\b", sentence.lower())  # ignore punctuation and capital
            words.update(clean_words)
        new_tokens = [word for word in words if word not in existing_vocab]
        return new_tokens

    def forward(
        self,
        sentences: List[str],
        target_labels: Tensor,
        return_hidden: bool = False,
        device=None,
        ):

        # Dynamically adds new tokens from input sentences before processing
        new_tokens = extract_new_tokens(sentences, self.tokenizer)
        if new_tokens:
            num_added_toks = self.tokenizer.add_tokens(new_tokens)
            print(f"Added {num_added_toks} new tokens to discriminator: {new_tokens}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        inputs = self.tokenizer(sentences,
            truncation=self.truncation, 
            padding=self.padding, 
            max_length=self.max_seq_length,
            return_tensors="pt")

        inputs["labels"] = target_labels
        inputs = inputs.to(device)
        output = self.model(**inputs, output_hidden_states=return_hidden)

        return output, output.loss
    
    def save_model(
        self, 
        path: Union[str]
        ):

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
