from transformers import AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import pickle


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained('/Users/tom/Downloads/bert-cased')
        self.classifier = torch.nn.Linear(768 * 2, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, bug_ids, fix_ids, bug_attention_mask, fix_attention_mask):
        bug_embedding, fix_embedding = self.encoder(bug_ids, attention_mask=bug_attention_mask).last_hidden_state[:, 0,
                                       :], self.encoder(fix_ids, attention_mask=fix_attention_mask).last_hidden_state[:,
                                           0, :]
        patch_embedding = torch.cat((bug_embedding, fix_embedding), dim=-1)
        return self.softmax(self.classifier(patch_embedding))


# 定义数据集类，用于处理文本数据
class TextDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


train_dataloader = DataLoader(TextDataset('./data/train_fold_0.pickle'), batch_size=32, shuffle=True)
test_dataloader = DataLoader(TextDataset('./data/test_fold_0.pickle'), batch_size=1, shuffle=False)
model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


# 定义训练函数
def train_epoch():
    global model
    model = model.train()
    losses = []
    for d in train_dataloader:
        bug_ids = d["bug_ids"].to(device)
        fix_ids = d["fix_ids"].to(device)
        bug_attention_mask = d["bug_attention_mask"].to(device)
        fix_attention_mask = d["fix_attention_mask"].to(device)
        labels = d["label"].to(device)

        logits = model(
            bug_ids=bug_ids,
            fix_ids=fix_ids,
            bug_attention_mask=bug_attention_mask,
            fix_attention_mask=fix_attention_mask,
        )

        loss_function = torch.nn.CrossEntropyLoss().to(device)
        loss = loss_function(logits, labels)
        print(loss)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# 定义评估函数
def eval_model():
    global model
    model = model.eval()
    preds = []
    labels = []
    probs = []
    patches = []
    with torch.no_grad():
        for d in test_dataloader:
            bug_ids = d["bug_ids"].to(device)
            fix_ids = d["fix_ids"].to(device)
            bug_attention_mask = d["bug_attention_mask"].to(device)
            fix_attention_mask = d["fix_attention_mask"].to(device)
            label = d["label"]
            labels.extend(label.tolist())
            patches.extend(d["patch"])

            logits = model(
                bug_ids=bug_ids,
                fix_ids=fix_ids,
                bug_attention_mask=bug_attention_mask,
                fix_attention_mask=fix_attention_mask,
            )
            prob = logits.detach().cpu()[:, 0].tolist()
            probs.extend(prob)

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred)
    # 将0视为正样本，因此需要调整y_true和y_pred的值
    y_true_adjusted = [1 if x == 0 else 0 for x in labels]
    y_pred_adjusted = [1 if x == 0 else 0 for x in preds]

    # 计算指标
    accuracy = accuracy_score(y_true_adjusted, y_pred_adjusted)
    precision = precision_score(y_true_adjusted, y_pred_adjusted)
    recall = recall_score(y_true_adjusted, y_pred_adjusted)
    F1 = f1_score(y_true_adjusted, y_pred_adjusted)

    # 如果y_pred是概率值，可以使用roc_auc_score计算AUC
    # y_pred_prob = [0.9, 0.8, 0.3, 0.75, 0.6, 0.85, 0.4, 0.95, 0.55, 0.65]
    AUC = roc_auc_score(y_true_adjusted, probs)
    return accuracy, precision, recall, F1, AUC, preds, patches


best_acc = -1
epochs = 20
all_preds = []

for epoch in range(epochs):
    train_loss = train_epoch()
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}')

    acc, prec, rec, f1, auc, preds, patches = eval_model()
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}, AUC: {auc}')
    if acc > best_acc:
        best_acc = acc
        all_preds = list(zip(patches, preds))
print(all_preds)
