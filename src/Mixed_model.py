from transformers import AutoModel
import torch.nn as nn
from kan import KAN
import torch

class MIXEDModel(nn.Module):
    def __init__(self, max_lenght:int, model_name:str, device:str):
        super(MIXEDModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.mlp = nn.Linear(max_lenght*768, max_lenght*96, device=device)
        self.kan = KAN(width=[max_lenght*96, 3], device=device)

    def forward(self, tokens):
        batch_size = tokens['input_ids'].shape[0]
        predictions = self.model(**tokens)
        cls_output = predictions.last_hidden_state

        data = cls_output.view(batch_size, -1)
        data = self.mlp(data)
        return self.kan(data)

# MAIN
if __name__ == '__main__' :
    import torch
    from dataset import TextDataset
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lenght = 128

    print("loading data...")
    data = load_dataset("tommasobonomo/sem_augmented_fever_nli",cache_dir="../data/sem_augmented_fever", trust_remote_code=True)
    print("data loaded")

    print("loading model...")
    model = MIXEDModel(device, lenght, "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print("model loaded")

    print("tokenizing data...")
    i = 0
    KAN_dataset = []
    for data in data['validation']:
        tokens = tokenizer(
                data['premise'], 
                data['hypothesis'], 
                truncation=True, 
                padding='max_length',
                max_length=lenght,
                return_tensors='pt'
                ).to(device)
        
        label_map = {
        'ENTAILMENT': 0,
        'CONTRADICTION': 1,
        'NEUTRAL': 2
        }
        # Map the labels
        label = label_map[data['label']]

        tokens.update({'labels': label})
        KAN_dataset.append(tokens)
        i += 1
        if not i%1000: print("step:",i)
    dataset = TextDataset(KAN_dataset)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print('data processed')
    for step, data in enumerate(dataloader):
        inputs = {k: data[k] for k in ['input_ids', 'attention_mask']}
        predictions = model(inputs)

        print(predictions)