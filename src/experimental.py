from transformers import AutoModel
import torch.nn as nn
from kan import KAN

class KANModel(nn.Module):
    def __init__(self, device:str, max_lenght:int, model_name:str = "microsoft/mdeberta-v3-base"):
        super(KANModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.kan = KAN(width=[max_lenght*768, 3], device=device)

    def forward(self, tokens):
        batch_size = tokens['input_ids'].shape[0]
        predictions = self.model(**tokens)
        cls_output = predictions.last_hidden_state

        data = cls_output.view(batch_size, -1)

        return self.kan(data)

# MAIN
if __name__ == '__main__' :
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = load_dataset("fever","v2.0",cache_dir="../data/fever", trust_remote_code=True)
    print("data loaded")
    model = KANModel(device, 32, "microsoft/mdeberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    print("model loaded")
    dataloader = DataLoader(data['validation'], batch_size=128, shuffle=True)

    for step, inputs in enumerate(dataloader):
        inputs = inputs['claim']
        tokens = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt',max_length=32)
        tokens.to(device)
        predictions = model(tokens)

        print(predictions)
