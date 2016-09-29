
class Corpus(object):
    """
    Abstract class of Corpus. List out the interface function that needs to implemented
    """

    def get_corpus(self, *args, **kwargs):
        """
        :return: a dict of train, dev and test. Each element's format is domain dependent
        """
        raise NotImplementedError("get_corpus is required")