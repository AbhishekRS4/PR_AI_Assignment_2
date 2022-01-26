
import sys
from sklearn.metrics import f1_score

tree = [[135, 0, 0, 0, 1], [0, 136, 2, 1, 2], [1, 4, 294, 1, 0], [0, 1, 1, 144, 0], [0, 4, 0, 1, 73]]

y_true = [0 for x in range(sum(tree[0]))] + [1 for x in range(sum(tree[1]))]+  [2 for x in range(sum(tree[2]))]+ [3 for x in range(sum(tree[3]))] +[4 for x in range(sum(tree[4]))] 

y_pred = []
for row in tree:
    for i, n in enumerate(row):
        for x in range(n):
            y_pred.append(i)

print(f1_score(y_true, y_pred, average="weighted"))
