from transformers import AutoModel
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, device:str, lenght:int, model_name:str = "microsoft/deberta-v3-base"):
        super(BaseModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.linear = nn.Linear(lenght*768, 3, device=device)
        self.lenght = lenght

    def forward(self, tokens):
        batch_size = tokens['input_ids'].shape[0]
        predictions = self.model(**tokens)
        cls_output = predictions.last_hidden_state[:,(-1,-self.lenght),:]

        return self.linear(cls_output.view(batch_size, -1))

# MAIN
if __name__ == '__main__' :
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = load_dataset("fever","v2.0",cache_dir="../data/fever", trust_remote_code=True)
    print("data loaded")
    model = BaseModel(device, 32, "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    print("model loaded")
    dataloader = DataLoader(data['validation'], batch_size=128, shuffle=True)

    model.eval()

    for step, inputs in enumerate(dataloader):
        inputs = inputs['claim']
        tokens = tokenizer(inputs, padding=True, return_tensors='pt')
        tokens.to(device)
        with torch.no_grad():
           predictions = model(tokens)

        print(predictions)
