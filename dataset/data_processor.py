# ======================================= #
# ----------- Data4BertModel ------------ #
# ======================================= #
import os
from dataset.utils import SentenceProcessor


class IMDB2_A(SentenceProcessor):
    NAME = 'IMDB2-A'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'imdb2', 'a', 'train-a.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'imdb2', 'a', 'dev-a.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'imdb2', 'a', 'test-a.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class IMDB2_B(SentenceProcessor):
    NAME = 'IMDB2-B'
    NUM_CLASSES = 10

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'imdb2', 'b', 'train-b.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'imdb2', 'b', 'dev-b.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'imdb2', 'b', 'test-b.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP2_A(SentenceProcessor):
    NAME = 'YELP2-A'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp2', 'a', 'train-a.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp2', 'a', 'dev-a.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp2', 'a', 'test-a.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class YELP2_B(SentenceProcessor):
    NAME = 'YELP2-B'
    NUM_CLASSES = 5

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'yelp2', 'b', 'train-b.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'yelp2', 'b', 'dev-b.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'yelp2', 'b', 'test-b.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_sent_doc(self):
        train = self._creat_sent_doc(self.d_train)
        dev = self._creat_sent_doc(self.d_dev)
        test = self._creat_sent_doc(self.d_test)
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


from dataset.utils import pd
class GDRD_A(SentenceProcessor):
    NAME = 'GDRD-A'
    NUM_CLASSES = 5

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=1, encoding="utf-8", sep='\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # [ user, product, review, label]
            document = list([str(pd_reader[0][i]), str(pd_reader[1][i]), pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'gdrd', 'a', 'goodreads_train_A.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'gdrd', 'a', 'goodreads_dev_A.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'gdrd', 'a', 'goodreads_test_A.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class GDRD_B(SentenceProcessor):
    NAME = 'GDRD-B'
    NUM_CLASSES = 5

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=1, encoding="utf-8", sep='\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # [ user, product, review, label]
            document = list([str(pd_reader[0][i]), str(pd_reader[1][i]), pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'gdrd', 'b', 'goodreads_train_B.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'gdrd', 'b', 'goodreads_dev_B.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'gdrd', 'b', 'goodreads_test_B.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class PPR_A(SentenceProcessor):
    NAME = 'PPR-A'
    NUM_CLASSES = 5

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=1, encoding="utf-8", sep='\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # [ user, product, review, label]
            document = list([str(pd_reader[0][i]), str(pd_reader[1][i]), pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'ppr', 'a', 'ppr_train_A.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'ppr', 'a', 'ppr_dev_A.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'ppr', 'a', 'ppr_test_A.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)


class PPR_B(SentenceProcessor):
    NAME = 'PPR-B'
    NUM_CLASSES = 5

    def _read_file(self, dataset):
        pd_reader = pd.read_csv(dataset, header=None, skiprows=1, encoding="utf-8", sep='\t', engine='python')
        documents = []
        for i in range(len(pd_reader[0])):
            # [ user, product, review, label]
            document = list([str(pd_reader[0][i]), str(pd_reader[1][i]), pd_reader[3][i], pd_reader[2][i]])
            documents.append(document)
        return documents

    def __init__(self, data_dir='corpus'):
        self.d_train = self._read_file(os.path.join(data_dir, 'ppr', 'b', 'ppr_train_B.tsv'))
        self.d_dev = self._read_file(os.path.join(data_dir, 'ppr', 'b', 'ppr_dev_B.tsv'))
        self.d_test = self._read_file(os.path.join(data_dir, 'ppr', 'b', 'ppr_test_B.tsv'))

    def get_documents(self):
        train = self._create_examples(self.d_train, 'train')
        dev = self._create_examples(self.d_dev, 'dev')
        test = self._create_examples(self.d_test, 'test')
        return tuple([train, dev, test])

    def get_attributes(self):
        return self._get_attributes(self.d_train, self.d_dev,
                                    self.d_test)  # tuple(attributes) rather tuple(users, products)