import matplotlib.pyplot as plt
import numpy as np


class Analyst:

    def __init__(self, accuracy_list, loss_list, iteration):
        self.accuracy_list = accuracy_list
        self.loss_list = loss_list
        self.iteration = iteration
        # self.draw()

    def draw(self):
        x1 = range(0, self.iteration)
        x2 = range(0, self.iteration)
        y1 = self.accuracy_list
        y2 = self.loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Test accuracy vs. epoches')
        plt.ylabel('Test accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Test loss vs. epoches')
        plt.ylabel('Test loss')
        plt.show()
        plt.savefig("accuracy_loss.jpg")

    def average_score(self, num=3, _scores=[]):
        new_scores = []
        for i in range(0, len(_scores), num):
            sum = 0
            for j in range(0, num):
                if i + j < len(_scores):
                    sum += _scores[i + j]
            sum /= num
            new_scores.append(sum)
        self.iteration = len(new_scores)
        self.accuracy_list = len(new_scores) * [1]
        self.loss_list = new_scores

        self.draw()

    def inv_hit_rate(self, _scores):
        hit_num = 0
        hit_rates = []
        for i in range(0, len(_scores)):
            if _scores[i] != -10:
                hit_num += 1
            hit_rates.append(hit_num / (i + 1))
        # self.iteration = len(hit_rates)
        # self.accuracy_list = len(_scores) * [1]
        # self.loss_list = hit_rates
        return hit_rates

    def inv_draw_two_hit_rate(self, scores1, scores2):
        hits1 = self.inv_hit_rate(scores1)
        hits2 = self.inv_hit_rate(scores2)
        x2 = range(0, len(hits2))
        x1 = range(0, len(hits1))
        plt.plot(x1, hits1, "r",label="rl")
        plt.plot(x2, hits2, "g",label="random")
        plt.legend()

        plt.title("不变式命中率")
        plt.xlabel("epochs")
        plt.ylabel("hit rate")
        plt.show()


if __name__ == "__main__":
    # scores = [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 5, -1, -10, -10, -1, -10, -10,
    #           5, 5, -10, -1, 5, -10, -10, -10, -10, -1, -1, -10, -10, -10, -10, -10, -10, -10, 5] toy_consensus_epr

    # scores1 = [-10, -10, -10, -10, -1, -1, -10, -10, -10, -10, -1, -10, -10, -10, -10, -10, -10, -10, -10, -10,
    #            -10, -1, -10, -10, -10, -10, -10, -10, -10, -10, -1, -10, -10, -10, -10, -1, -10, -10, -10, -1, -10,
    #            -10, -10, -10, -10, -1, -10, -10, -1, -10, -10, -10, -1, -1, -10, -10, -10, -10, -10, -10, -10, -1,
    #            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
    #            -1, 5, -1, -1, -10, -1, -10, -10, -10, -1, -10, -10, -1, -10, -10, -1, -1, -10, -1, -1, -1, -10,
    #            -1]  # toy_consensus_forall
    #
    # scores2 = [-10, -10, -10, -10, -10, 5, -10, -10, -10, -10, 5, -10, -10, -10, -10, -10, -1, -1, -10, 5, -10, -10,
    #            -10, -10, -10, -10, -10, -10, -10, -1, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
    #            -10, -10, -10, -10, -10, -10, -10, 5, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -1, -10,
    #            -10, -10, -10, -1, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
    #            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -1, -10, -10, -10, -10, -10]

    scores2 = [-10, 5, -10, -1, -10, -10, -10, -10, -10, -10, -10, -10, -10, 5, -10, -10, -10, 5, -10, -10, -10, -10,
              -10, -10, -10, 5, -10, -10, 5, -10, -10, -10, -10, 5, -10, -10, -10, -10, -10, -10, 5, -10, 5, -10,
              -10, -10, -10, -10, -10, 5, -10, 5, -10, -10, -10, -10, 5, -10, -10, -10, -10, -10, -10, -10, -10,
              -10, -10, -10, -10, -10, -10, -10, 5, -10, -1, -10, -10, -10, -10, -10, -10, 5, -10, -1, -10, -10,
              -10, -10, -10, -10, 5]# lockserv random

    scores1 = [-10, -10, -10, -10, -10, 5, -10, -10, -10,
              -10, -10, -10, -10, -10, -10, -10, -10, 5, 5, -10, -10, 5, -10, -10, -10, -10, -10, -10, -10, -10,
              -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 5, -10, -1, -1, -10, -10, 5, -1, -1, -10, -1,
              -10, -10, -1, -10, -10, -10, -10, -10, -1, 5, -10, -10, -10, -10, 5, -10, -1, -1, -10, -10, -10, -10,
              -10, -10, -10, -1, -1, -1, -10, -10] #lockserv
    analyst = Analyst([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], 5)

    analyst.inv_draw_two_hit_rate(scores1, scores2)
