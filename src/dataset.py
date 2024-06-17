from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.input_ids = [item['input_ids'].squeeze(0) for item in tokenized_texts]
        self.attention_masks = [item['attention_mask'].squeeze(0) for item in tokenized_texts]
        self.labels = [item['labels'] for item in tokenized_texts]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }
    
# MAIN
if __name__ == '__main__' :
    ...
