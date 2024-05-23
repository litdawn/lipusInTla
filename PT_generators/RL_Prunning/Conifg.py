class Config:
    use_self_generate = False
    SELECT_AN_ACTION = 0
    SET_AN_VALUE = 1

    SIZE_EXP_NODE_FEATURE = 128
    SIZE_PCA_NUM = 50

    MAX_DEPTH = 150
    BEST = False

    CONTINUE_TRAINING = True

    LinearPrograms = ["Problem_L" + str(i) for i in range(1, 134)]
    NonLinearPrograms = ["Problem_NL" + str(i) for i in range(1, 31)]

    LearningRate = 0.01

    generate_time = {
        "very": 5.0,
        "little": 1.0
    }
    min_num_conjuncts = 2
    max_num_conjuncts = 3

    seed_num = 3

    DISCOUNTER = 0.5

    ALOSS = 0.1
    SLOSS = 0.2


config = Config()
