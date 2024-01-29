class SequenceDataPoint:
    def __init__(self):
        """
        Create a single data point, which is a single unit of
        item in the dataset. All the attributes will be set
        by DataFormatter
        """
        self.file_index = ""
        self.function1 = ""
        self.function2 = ""
        self.function_vec = ""  # All
        self.function_vec1 = ""  # Bug
        self.function_vec2 = ""  # Repair
        self.tgt = ""  # Target of the Function
        self.tgt_vec = ""

