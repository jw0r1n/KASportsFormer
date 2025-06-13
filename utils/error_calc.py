import torch
import numpy as np


def mpjpe_calc(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    assert predicted.shape == target.shape, 'shape is unaligned (mpjpe_calc)'
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)  # (27, 17, 3) -> (27, 17) -> (27,)


def jpe_calc(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    assert predicted.shape == target.shape, 'shape unaligned (jpe_calc)'
    return np.linalg.norm(predicted - target, axis=len(target.shape) - 1)  # (27, 17)


def acc_error_calc(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    accel_ground_truth = target[:-2] - 2 * target[1: -1] + target[2:]  # (25, 17, 3)
    accel_predict = predict[:-2] - 2 * predict[1: -1] + predict[2:]
    normed = np.linalg.norm(accel_predict - accel_ground_truth, axis=2)  # (25, 17)
    return np.mean(normed, axis=1)  # (25, )

def p_mpjpe_calc(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    assert  predict.shape == target.shape, 'shape is not aligned (p_mpjpe_calc)'
    muX = np.mean(target, axis=1, keepdims=True)  # (27, 1, 3)
    muY = np.mean(predict, axis=1, keepdims=True)
    X0 = target - muX
    Y0 = predict - muY
    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))  # (27, 1, 1) -> (27, 1, 1)
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))
    X0 /= normX   # 减去平均再除以norm
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)  # (27, 3, 3)
    U, s, Vt = np.linalg.svd(H)    # U s Vt (27, 3, 3) (27, 3) (27, 3, 3)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))  # (27, 3, 3)

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))    # det (27,) -> expand (27, 1) -> (27, 1)
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)  # (27, 1, 1)
    a = tr * normX / normY  # Scale  (27, 1, 1)
    t = muX - a * np.matmul(muY, R)  # Translation (27, 1, 3)
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predict, R) + t  # (27, 17, 3)
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)  # (27, 17) -> (27, )




if __name__ == '__main__':
    temp1 = np.random.random((27, 17, 3))
    temp2 = np.random.random((27, 17, 3))
    # print(temp3.shape)
    # temp4 = np.sum(temp3 ** 2, axis=(1, 2), keepdims=True)
    # temp4 = np.sqrt(temp4)
    # print(temp4.shape)
    normX = np.sqrt(np.sum(temp1 ** 2, axis=(1, 2), keepdims=True))  # (27, 1, 1) -> (27, 1, 1)
    normY = np.sqrt(np.sum(temp2 ** 2, axis=(1, 2), keepdims=True))

    temp3 = np.matmul(temp1.transpose(0, 2, 1), temp2)
    U, s, Vt = np.linalg.svd(temp3)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY
    t = np.random.random((27, 17, 3))
    temp4 = a * np.matmul(temp1, R) + t
    interval = a * np.matmul(temp1, R) + t
    interval2 = np.linalg.norm(interval - temp2, axis=len(temp2.shape) - 1)
    interval3 = np.mean(interval2, axis=1)
    print(interval3.shape)






