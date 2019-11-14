f = open('in.txt')

tp=0
tn=0
fp=0
fn=0

precision=0
recall=0
accuracy=0

while True:
    line = f.readline()

    if not line: break
    line=line.split()
    t_cond=line[5]#test case condition
    t_pred=line[2]#test case prediction
    
    #state 1: anomaly, state 0: normal
    #print(t_cond, t_pred)

    if t_cond == '1' and t_pred == '1': 
        tp+=1 
    elif t_cond=='1' and t_pred=='0': 
        fp=fp+1
    elif t_cond=='0' and t_pred=='0': 
        tn=tn+1
    elif t_cond=='0' and t_pred=='1': 
        fn=fn+1

accuracy=round((tp+tn)/(tp+tn+fp+fn),2)
recall=round(tp/(tp+fp),2)
precision=round(tp/(tp+fn),2)

print (tp, fp, tn, fn, tp+tn+fp+fn)
print(accuracy, recall, precision)
