class Lemma2Candidate:
    # def __init__(self):
    #     self.a = "a"
    @staticmethod
    def select(reward_list: dict, lemma_num=5):
        reward_list = sorted(reward_list.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        ans = dict()
        for name, reward in reward_list:
            ans.update({name: reward})
        return ans
