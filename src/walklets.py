import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from helper import walk_transformer, create_graph
from walkers import FirstOrderRandomWalker, SecondOrderRandomWalker

class WalkletMachine:
    def __init__(self, args):

        self.args = args
        self.graph = create_graph(self.args.input)
        if self.args.walk_type == "first":
            self.walker = FirstOrderRandomWalker(self.graph, args)
        else:
            self.walker = SecondOrderRandomWalker(self.graph, False, args)
            self.walker.preprocess_transition_probs()
        self.walks = self.walker.do_walks()
        del self.walker
        self.create_embedding()
        self.save_model()

    def walk_extracts(self, length):

        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_embedding(self, model):

        embedding = []
        for node in range(len(self.graph.nodes())):
            embedding.append(list(model[str(node)]))
        embedding = np.array(embedding)
        return embedding

    def create_embedding(self):

        self.embedding = []
        for index in range(1, self.args.window_size+1):
            print("\nOptimization round: "+str(index)+"/"+str(self.args.window_size)+".")
            print("Creating documents.")
            clean_documents = self.walk_extracts(index)
            print("Fitting model.")

            model = Word2Vec(clean_documents,
                             size=self.args.dimensions,
                             window=1,
                             min_count=self.args.min_count,
                             sg=1,
                             workers=self.args.workers)

            new_embedding = self.get_embedding(model)
            self.embedding = self.embedding + [new_embedding]
        self.embedding = np.concatenate(self.embedding, axis=1)

    def save_model(self):

        print("\nModels are integrated to be multi scale.\nSaving to disk.")
        self.column_names = ["x_" + str(x) for x in range(self.embedding.shape[1])]
        self.embedding = pd.DataFrame(self.embedding, columns=self.column_names)
        self.embedding.to_csv(self.args.output, index=None)
