from sklearn.naive_bayes import MultinomialNB


class NaiveBayesModelFeature:
    def __init__(self, config):
        self.config = config
        self.trained_model = None

    def train(self, train_x, train_y):
        assert self.trained_model is None
        model = MultinomialNB().fit(train_x, train_y)

        self.trained_model = model

    def val(self, val_x, val_y):
        assert self.trained_model is not None
        preds = self.trained_model.predict_log_proba(val_x)
        return preds
