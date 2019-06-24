import numpy as np


class DistributedRepresentations:
    """Distributed Represendations of the words.

    Args:
        words (list): List of words
        vectors (numpy.array): Vectors encoded words

    Attributes:
        vecs (numpy.array): Vectors encoded words
        words (list): List of words
    """
    def __init__(self, words, vectors):
        self.words = words
        self.vecs = vectors

    @property
    def normalized_vecs(self):
        return self.vecs / \
            np.linalg.norm(self.vecs, axis=1).reshape(
                [self.vecs.shape[0], 1],
            )

    def inspect(self):
        rels = np.dot(self.normalized_vecs, self.normalized_vecs.T)
        printoptions = np.get_printoptions()
        np.set_printoptions(linewidth=200, precision=6)
        for word, vec in zip(self.words, rels):
            print(f'{word + ":":8s} {vec}')
        np.set_printoptions(**printoptions)

    def cos_similarity(self, x, y, eps=1e-8):
        return np.dot(x, y) / (np.linalg.norm(x) + eps) \
            / (np.linalg.norm(y) + eps)

    def words_similarity(self, word1, word2, eps=1e-8):
        x, y = [self.vecs[i]
                for i in [self.words.index(word) for word in [word1, word2]]]
        return self.cos_similarity(x, y, eps=eps)

    def most_similar(self, word, top=5):
        try:
            word_id = self.words.index(word)
        except ValueError:
            print(f"'{word}' is not found.")
            return
        print(f'\n[query]: {word}')
        word_vec = self.vecs[word_id]
        similarity = [[w, self.cos_similarity(word_vec, self.vecs[i])]
                      for i, w in enumerate(self.words) if i != word_id]
        similarity.sort(key=lambda sim: sim[1], reverse=True)
        for s in similarity[:top]:
            print(f' {s[0]}: {s[1]}')

    def analogy(self, a, b, c, top=5, answer=None):
        try:
            a_vec, b_vec, c_vec = \
                self.vecs[[self.words.index(word) for word in (a, b, c)]]
        except ValueError as err:
            print(err)
            return

        print(f'{a}:{b} = {c}:?')
        query_vec = b_vec - a_vec + c_vec
        if answer is not None:
            try:
                answer_id = self.words.index(answer)
                print(
                    f'  ==> {answer}: '
                    f'{self.cos_similarity(self.vecs[answer_id], query_vec)}'
                )
            except ValueError as err:
                print(err)
        similarity = [[w, self.cos_similarity(query_vec, self.vecs[i])]
                      for i, w in enumerate(self.words)]
        similarity.sort(key=lambda sim: sim[1], reverse=True)
        for s in similarity[:top]:
            print(f'  {s[0]}: {s[1]}')
        print()
