class ValueTransformer:
    def __init__(self, type="diff") -> None:
        # log1p, diff, sqrt
        self.type = type

    def transform(self, x):
        return x

    def inverse_transform(self, y):
        return y
