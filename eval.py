#!python3
import sys
with open(sys.argv[1], "r") as true:
    with open(sys.argv[2], "r") as tested:
        T = 0
        F = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for gold, pred in zip(true, tested):
            gold, pred = gold.strip().split("\t")[-1], pred.strip().split("\t")[-1]
            if gold.count("1") > 1:
                continue
            if all(g == p for g, p in zip(gold.split(), pred.split())):
                T += 1
            else:
                F += 1
                #print(gold, pred)

            for g, p in zip(gold.split(), pred.split()):
                g, p = int(g in {"B", "S"}), int(p in {"B", "S"})
                if g == 1 and p == 1:
                    tp += 1
                elif g == 1 and p == 0:
                    fn += 1
                elif g == 0 and p == 1:
                    fp += 1
                elif g == 0 and p == 0:
                    tn += 1
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        F_measure = (2*precision*recall)/(precision + recall)
        print(sys.argv[2] + "&" + " \%&".join([str(round(100 * x, 1)) for x in [T/(T+F),(tp+tn)/(tp + fp + tn + fn), precision, recall, F_measure]]) + " \% \\\\")
        #results_sk.tsvprint(tp, fp, tn, fn)
