import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, cohen_kappa_score, precision_recall_curve, average_precision_score, accuracy_score

manual1_base_directory = "../Data/DRIVE/test/1st_manual/"
test_base_directory = "../Data/DRIVE/tmp/"
manual1 = []
test = []

prior_threshold = True

# load images
for i in range(1,21):
    if i < 10:
        img = misc.imread(manual1_base_directory + "0" + str(i) + "_manual1.gif")
        manual1.extend(img.reshape(img.shape[0] * img.shape[1]))
        img = misc.imread(test_base_directory + "0" + str(i) + "_test.png")
        test.extend(img.reshape(img.shape[0] * img.shape[1]))
    else:
        img = misc.imread(manual1_base_directory + str(i) + "_manual1.gif")
        manual1.extend(img.reshape(img.shape[0] * img.shape[1]))
        img = misc.imread(test_base_directory + str(i) + "_test.png")
        test.extend(img.reshape(img.shape[0] * img.shape[1]))

# turn into 0/1 classification
for i in range(len(manual1)):
    if manual1[i] != 0:
        manual1[i] = 1


# obtain maximum average accuracy and threshold
accuracy = []
thresholds = []

if prior_threshold:
    start = 61800
    end = 61900
    step = 100
else:
    start = 60000
    end = 65535
    step = 100

for j in xrange(start,end,step):
    y_pred = np.copy(test)
    print "Testing current threshold: " + str(j)
    for i in range(len(test)):
        if test[i] >= j:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    accuracy.append( accuracy_score(manual1, y_pred))
    thresholds.append(j)
threshold = thresholds[accuracy.index(max(accuracy))]
print "maximum average accuracy = " + str(max(accuracy))
print "threshold = " + str(threshold)
y_pred = np.copy(test)
for i in range(len(test)):
    if test[i] >= threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

# obtain kappa
print "kappa = " + str(cohen_kappa_score(y_pred, manual1))

# obtain probability confidence scores
y_prob = np.copy(test).astype(float)
for idx, pixel in enumerate(test):
    y_prob[idx] = y_prob[idx] / 65535

# AUROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(manual1, y_prob)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='black',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

## precision-recall curve
precision, recall, _ = precision_recall_curve(manual1, y_prob)
average_precision = average_precision_score(manual1, y_prob)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
                                                               average_precision))
plt.show()
