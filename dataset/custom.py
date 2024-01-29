from bases.base_dataset import BaseDataset


class Custom(BaseDataset):
    def __init__(self, config, name="CustomDataset", running_mode='train'):
        """
        Initiate the Dataset of Online Judge Dataset
        :param config: Configuration file
        """
        BaseDataset.__init__(self, name, config, running_mode)