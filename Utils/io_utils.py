import csv
import numpy as np

def index_to_parameter(i):
    X = []
    num_freq = 3
    num_field = 5
    num_phase_z = 4
    num_phase_y = 4
    total = num_freq * num_freq * num_freq * num_field * num_field * num_field * num_phase_z * num_phase_y 
    num = i
    total = total / num_freq
    fx = num // total
    num = num % total
    X.append(3+2*fx)

    total = total / num_freq
    fy = num // total
    num = num % total
    X.append(3+2*fy)

    total = total / num_freq
    fz = num // total
    num = num % total
    X.append(3+2*fz)

    total = total / num_field
    Bx = num // total
    num = num % total
    X.append(2*Bx)

    total = total / num_field
    By = num // total
    num = num % total
    X.append(2*By)

    total = total / num_field
    Bz = num // total
    num = num % total
    X.append(2*Bz)

    total = total / num_phase_y
    phase_y = num // total
    num = num % total
    X.append(phase_y)

    total = total / num_phase_z
    phase_z = num // total
    num = num % total
    X.append(phase_z)
    return np.array(X)

def load_magnetic_field_para(verbose=False):
    X_1=[]
    X=[]
    y=[]
    with open ('input.csv') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        for row in f_csv:
            
            X_1.append(int(row[0]))
            y.append(int(row[1]))

    for i in X_1:
        num_freq = 3
        num_field = 5
        num_phase_z = 4
        num_phase_y = 4
        total = num_freq * num_freq * num_freq * num_field * num_field * num_field * num_phase_z * num_phase_y 
        num = i
        total = total / num_freq
        fx = num // total
        num = num % total
        X.append(3+2*fx)

        total = total / num_freq
        fy = num // total
        num = num % total
        X.append(3+2*fy)

        total = total / num_freq
        fz = num // total
        num = num % total
        X.append(3+2*fz)

        total = total / num_field
        Bx = num // total
        num = num % total
        X.append(2*Bx)

        total = total / num_field
        By = num // total
        num = num % total
        X.append(2*By)

        total = total / num_field
        Bz = num // total
        num = num % total
        X.append(2*Bz)

        total = total / num_phase_y
        phase_y = num // total
        num = num % total
        X.append(phase_y)

        total = total / num_phase_z
        phase_z = num // total
        num = num % total
        X.append(phase_z)
    X = np.array(X)
    X_1 = np.array(X_1)
    y = np.array(y)
    X = X.reshape(-1,8)
    if verbose:
        print(X, y)
    from sklearn import model_selection as ms
    X_train_pre, X_test_pre, y_train_pre, y_test_pre, = ms.train_test_split(X, y, test_size = 0.20, random_state=150)
    return X_train_pre, X_test_pre, y_train_pre, y_test_pre, X_1, y

if __name__ == "__main__":
    X_train_pre, X_test_pre, y_train_pre, y_test_pre, X_1, y = load_magnetic_field_para()
    np.save('./Data/X_train_pre', X_train_pre)
    np.save('./Data/X_test_pre', X_test_pre)
    np.save('./Data/y_train_pre', y_train_pre)
    np.save('./Data/y_test_pre', y_test_pre)
    np.save('./Data/X_1', X_1)
    np.save('./Data/y', y)