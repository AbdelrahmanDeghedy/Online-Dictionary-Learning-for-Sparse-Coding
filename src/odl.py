import numpy as np
from sklearn import linear_model

# Provides a progress bar
from tqdm import tqdm


class OnlineDictionaryLearning:
    def __init__(self, data):
        self.data = data
        self.numberOfObservations = len(self.data)

        self.losses = []
        self.cumulative_losses = []

        self.objective = []
        self.alphas = []
        self.observed = []

    def sample(self, data):
        while True :
            permutation = list(np.random.permutation(self.numberOfObservations))
            for idx in permutation:
                yield data[idx]


    def computeObjectiveFunction(self):
        cumulated_loss = np.cumsum(self.losses)
        self.objective = [cumulated_loss[i] / (i + 1) for i in range(len(cumulated_loss))]

    @staticmethod
    def compute_alpha(x, dictionary, regularizationParameter):
        reg = linear_model.LassoLars(alpha=regularizationParameter, normalize=True)
        reg.fit(X=dictionary, y=x)
        return reg.coef_

    @staticmethod
    def initialize_dictionary(dict_size, data_gen):
        return np.array([next(data_gen) for _ in range(dict_size)]).T

    # Algorithm 2 Dictionary Update.
    @staticmethod
    def update_dictionary(a, b, dictionary, dict_size):
        # For each column of the dictionary
        for j in range(dict_size) :
            u_j = dictionary[:, j] + (b[:, j] - np.matmul(dictionary, a[:, j])) / a[j, j]
            dictionary[:, j] = u_j / max([1, np.linalg.norm(u_j)])

        return dictionary

    # Algorithm 1 Online dictionary learning
    def learn(self, iterations, regularizationParameter, dict_size):
        sampledData = self.sample(self.data)

        a_prev = 0.01 * np.identity(dict_size)
        b_prev = 0
        d_prev = self.initialize_dictionary(dict_size, sampledData)

        for _ in tqdm(range(iterations)):
            x = next(sampledData)

            alpha = self.compute_alpha(x, d_prev, regularizationParameter)

            a_curr = a_prev + np.outer(alpha, alpha.T)
            b_curr = b_prev + np.outer(x, alpha.T)

            currDictionary = self.update_dictionary(a=a_curr, b=b_curr, dictionary=d_prev, dict_size=dict_size)

            a_prev = a_curr
            b_prev = b_curr
            d_prev = currDictionary

            self.computeObjectiveFunction()

            self.log(observation=x, dictionary=currDictionary, alpha=alpha)

        return currDictionary.T

    # Calculates the loss for each observation
    @staticmethod
    def one_loss(x, dictionary, alpha):
        return np.linalg.norm(x - np.matmul(dictionary, alpha), ord=2) ** 2
    
    # Calculates the cumulative (average) loss for all observations
    def cumulative_loss(self, dictionary):
        n_observed = len(self.observed)
        return sum([self.one_loss(self.observed[i], dictionary, self.alphas[i]) for i in range(n_observed)]) / n_observed

    def log(self, observation, dictionary, alpha):
        loss = self.one_loss(observation, dictionary, alpha)
        self.losses.append(loss)
        self.alphas.append(alpha)
        self.observed.append(observation)
        # self.objective.append(objective)
        self.cumulative_losses.append(self.cumulative_loss(dictionary))

