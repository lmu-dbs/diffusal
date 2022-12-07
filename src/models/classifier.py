import torch
from sklearn.metrics import accuracy_score


class Classifier:
    """
    Classifier abstraction to be used in the AL framework.
    Can use any classifier internally, e.g. pytorch modules,
    scikit-learn classifiers etc.
    """

    def __init__(self):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def predict_probs(self, data):
        raise NotImplementedError

    def get_test_accuracy(self, data):
        pred = self.predict_probs(data).argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        return int(test_correct.sum()) / int(data.test_mask.sum())


class TorchClassifier(Classifier):
    """
    Classifier abstraction using pytorch modules internally,
    e.g. MLP, GCN.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.lr = model.lr
        self.weight_decay = model.weight_decay

        self.best_accuracy = 0

    def get_embedding(self, data):
        return self.model.get_embedding(data)

    def train_intermediate(self, data, num_epochs=300, reset_params=False):
        # no validation set
        if reset_params:
            self.model.reset_parameters()

        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            # Train step
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = self.predict_probs(data)
            pred_max = pred.argmax(dim=1)
            test_correct = pred_max[data.test_mask] == data.y[data.test_mask]

    def train(self, data, max_epochs=1000, patience=25, reset_params=True):
        if reset_params:
            self.model.reset_parameters()

        # torch.manual_seed(0)

        self.best_accuracy = 0

        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        best_val_loss = float('inf')

        for epoch in range(max_epochs):
            # Train step
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                pred = self.predict_probs(data)
                pred_max = pred.argmax(dim=1)
                val_correct = pred_max[data.val_mask] == data.y[data.val_mask]
                val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch:5}: {val_loss:2.7f} (acc: {val_acc})')
                if val_acc >= best_val_acc or val_loss <= best_val_loss:
                    test_correct = pred_max[data.test_mask] == data.y[data.test_mask]
                    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
                    self.best_accuracy = test_acc

                    best_val_acc = max(val_acc, best_val_acc)
                    best_val_loss = min(val_loss, best_val_loss)
                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter == patience:
                        break

    def predict_probs(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data)
            out = torch.log_softmax(out, dim=1)
            out = torch.exp(out)
        return out


class SKLearnClassifier(Classifier):
    """
    Classifier abstraction for classifiers using scikit-learn 
    """

    def __init__(self, model):
        self.model = model

    def train(self, data, **kwargs):
        data = data.cpu()
        self.model.fit(data.x[data.train_mask], data.y[data.train_mask])

    def predict_probs(self, data):
        probs = self.model.predict_proba(data.x)
        return torch.from_numpy(probs)

    def get_test_accuracy(self, data):
        return accuracy_score(self.model.predict(data.x[data.test_mask]), data.y[data.test_mask])

    def get_val_accuracy(self, data):
        return accuracy_score(self.model.predict(data.x[data.val_mask]), data.y[data.val_mask])
