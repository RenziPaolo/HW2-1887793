from transformers import AutoModel
import torch.nn as nn
from kan import KAN

class BaseModel(nn.Module):
    def __init__(self, device:str, max_lenght:int, model_name:str = "microsoft/deberta-v3-base"):
        super(BaseModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.kan = KAN(width=[max_lenght*768, 3, 2], device=device)

    def forward(self, tokens):
        batch_size = tokens['input_ids'].shape[0]
        predictions = self.model(**tokens)
        cls_output = predictions.last_hidden_state

        data = cls_output.view(batch_size, -1)

        return self.kan()

# MAIN
if __name__ == '__main__' :
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = load_dataset("fever","v2.0",cache_dir="../data/fever", trust_remote_code=True)
    print("data loaded")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(device)
    kan = KAN(width=[32*768, 3, 2], device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    print("model loaded")
    dataloader = DataLoader(data['validation'], batch_size=128, shuffle=True)

    model.eval()

    for step, inputs in enumerate(dataloader):
        inputs = inputs['claim']
        tokens = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt',max_length=32)
        tokens.to(device)
        with torch.no_grad():
           predictions = model(**tokens)

        cls_output = predictions.last_hidden_state

        kan_data = cls_output.view(128, -1)

        predictions = kan() 

        print(predictions.shape)
