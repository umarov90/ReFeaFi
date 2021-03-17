sal_pred = None
for j in range(1000):
    sal = []
    for i in range(1001):
        seq = positive_set[j].copy().astype("float32")
        ind = np.where(seq[i])
        seq[i][ind] = seq[i][ind] + 0.01
        sal.append(seq)
    if sal_pred is None:
        sal_pred = np.asarray(brun(sess, input_x, y, sal, kr, in_training_mode))
    else:
        sal_pred = sal_pred + np.asarray(brun(sess, input_x, y, sal, kr, in_training_mode))