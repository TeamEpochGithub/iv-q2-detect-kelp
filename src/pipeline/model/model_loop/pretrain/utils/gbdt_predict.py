def predict(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # x has shape (B, C, H, W), transpose and reshape for catboost to (N, C)
    x_ = x.transpose((0, 2, 3, 1)).reshape((-1, x.shape[1]))

    # Predict and reshape back to (N, 1, H, W)
    pred = cbm.predict_proba(x_)[:, 1].reshape((x.shape[0], 1, x.shape[2], x.shape[3]))
    return np.concatenate([x, pred], axis=1)