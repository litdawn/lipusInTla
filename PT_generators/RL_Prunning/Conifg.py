class Config:
    SELECT_AN_ACTION = 0
    SET_AN_VALUE = 1

    SIZE_EXP_NODE_FEATURE = 128
    SIZE_PCA_NUM = 50

    MAX_DEPTH = 150
    BEST = False

    CONTINUE_TRAINING = True

    LinearPrograms = ["Problem_L" + str(i) for i in range(1, 134)]
    NonLinearPrograms = ["Problem_NL" + str(i) for i in range(1, 31)]

    LearningRate = 1e-6

    generate_time = {
        "very": 5.0,
        "little": 1.0
    }


config = Config()
