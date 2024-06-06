from dataset.custom import Custom
class DatasetFactory:
    name = "DatasetFactory"

    @staticmethod
    def get_dataset(config):
        """
        Factory Function to retrieve the necessary dataloader object
        A specific dataloader for a dataset
        :param config: Configuration object
        :return:
        """
        logger = config.logger
        dataset_name = config.dataset['name']
        logger.info("Dataset: %s" % dataset_name)
        if dataset_name == "custom":
            dataloader = Custom(config)
        else:
            raise SystemExit(Exception("Dataset Name %s is not found." % dataset_name))
        return dataloader
