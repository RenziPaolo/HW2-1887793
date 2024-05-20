import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab
from collections import Counter
from git import Repo

class JSONLDataset(Dataset):
    """
    PyTorch dataset class constructed from a dictionary, ideally read from JSON Lines (JSONL) file.
    """

    def __init__(self, test: bool, device: str, path:str, tokenizer) -> None: 
        """
        Initialize the dataset with a dictionary.

        Args:
        test (bool): True if the dataset is for testing, False otherwise.
        """

        self.data_dict = self.read_json_file(test)
        #
        self.device = device
        if tokenizer != None:
            self.tokenizedData = []
            for sample in self.data_dict:
                text = sample['text']  # Assuming 'text' is the key for the text data
                tokenizedText = [token.text for token in tokenizer(text)]  # Tokenizing text
                label = 0 if sample['label'] == 'si' else 1
                self.tokenizedData.append({'text': tokenizedText, 'label': label})
        self.Data = self.data_dict

    def read_json_file(self, test:  bool) -> list:
        """
        Read JSON data from a file and return it as a Python dictionary.

        Args:
        test (bool): True if the dataset is for testing, False otherwise.

        Returns:
        dict: The JSON data as a dictionary.
        """
        repo =  Repo(".", search_parent_directories=True)

        root_dir = repo.git.rev_parse("--show-toplevel")

        if test:
            file_path = root_dir+"/data/haspeede3-task1-test-data.jsonl"
        else:
            file_path = root_dir+"/data/haspeede3-task1-test-data.jsonl"
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = [json.loads(line.strip()) for line in file]
                return data_list
        except FileNotFoundError:
            print("File not found.")
            return None
        except json.JSONDecodeError:
            print("Invalid JSON format.")
            return None

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.Data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
        idx (int): Index of the sample.

        Returns:
        dict: Dictionary containing the sample data.
        """
        return self.Data[idx]

        
    def get_max_seq_len(self):
        return torch.max(torch.tensor([len(sample["input_ids"]) for sample in self.Data], dtype=torch.long)).item()



    
# MAIN
if __name__ == '__main__' :
    from transformers import BertTokenizer
    dataset = JSONLDataset(test=True, device='cuda', tokenizer=BertTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased"))
    print(dataset[0])
