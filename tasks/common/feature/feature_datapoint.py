class FeatureDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.features = ""
        self.tgt = ""  # Target of the Function
