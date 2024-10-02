import pandas as pd
import numpy as np
from scipy.stats import norm


def prepare_data(crowd, X, Y):
    TaskID_list = np.sort(crowd["TaskID"].unique())
    TaskID2i = {}
    i2TaskID = {}
    for i, ID in enumerate(TaskID_list):
        TaskID2i[ID] = i
        i2TaskID[i] = ID
    
    Is = [TaskID2i[ID] for ID in TaskID_list]
    X = X[Is]
    Y = Y[Is]
    return X, Y


def prepare_crowd(crowd, K):
    CrowdID = np.sort(crowd.CrowdID.unique())
    for i in CrowdID:
        tmp = crowd[crowd["CrowdID"] == i].CrowdLabel.value_counts().shape[0]
        if tmp < (K + 1):
            print(f"Drop [{i}]th crowd annotator's results because he/she only labels for {tmp} classes")
            crowd = crowd[crowd['CrowdID'] != i]
            print(f"Current Shape of Crowd Label Matrix:", crowd.shape)
            print(f"Current #Crowd Annotators:", crowd["CrowdID"].unique().shape[0], "\n")

    crowd = crowd.reset_index(drop=True)
    return crowd


def assign(crowd):
    ########### CrowdID and Index Map ###########
    CrowdID_list = np.sort(crowd["CrowdID"].unique())
    M = len(CrowdID_list)
    CrowdID2i = {}
    i2CrowdID = {}
    for i, ID in enumerate(CrowdID_list):
        CrowdID2i[ID] = i
        i2CrowdID[i] = ID
    ########### TaskID and Index Map ###########
    TaskID_list = np.sort(crowd["TaskID"].unique())
    n = len(TaskID_list)
    TaskID2i = {}
    i2TaskID = {}
    for i, ID in enumerate(TaskID_list):
        TaskID2i[ID] = i
        i2TaskID[i] = ID

    # Assigment Matrix & Crowd Label Matrix (n, M)
    A1 = np.zeros((n, M))    # Assigment Matrix
    AY1 = - np.ones((n, M))  # Crowd Label Matrix
    for i in range(crowd.shape[0]):
        row = TaskID2i[crowd.loc[i, "TaskID"]]          # which instance / task
        col = CrowdID2i[crowd.loc[i, "CrowdID"]]        # which crowd annotator / worker
        try:
            A1[row, col] = 1                            # assignement matrix
            AY1[row, col] = crowd.loc[i, "CrowdLabel"]  # crowd label matrix
        except:
            print(i, row, col)

    alpha = A1.mean(axis=0) # Assigment Probability per Crowd Annotator
    return A1, AY1, alpha


def transform(X):
    n = X.shape[0]
    p = X.shape[1]
    X_DF = pd.DataFrame(X)
    for j in range(p):
        X_DF[j] = X_DF[j].rank() / (n + 1)
        X_DF[j] = norm.ppf(X_DF[j])
    X = np.array(X_DF)
    return X

def add_constant(X):
    n = X.shape[0]
    p = X.shape[1]
    mat = np.ones((n, p + 1))
    mat[:, 1:] = X
    return mat