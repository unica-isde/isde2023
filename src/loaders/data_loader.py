from abc import ABC, abstractmethod


class DataLoader(ABC):
    """
    Abstract interface to implement data loaders
    """

    @abstractmethod
    def load_data(self):
        raise NotImplementedError("load_data not implemented.")


# class Child(DataLoader):
#    def __init___(self):
#        pass
#
#    def load_data(self):
#        super().load_data()


# my_instance = Child()
# my_instance.load_data()
