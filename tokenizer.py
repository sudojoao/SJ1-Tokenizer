from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()

files = [f"data/prompts.csv" for split in ["test", "train", "valid"]]

tokenizer.train(files, trainer)

tokenizer.save("data/tokenizer-lib.json")

tokenizer = Tokenizer.from_file("data/tokenizer-lib.json")