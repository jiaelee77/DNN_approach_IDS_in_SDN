#f = open('input.txt')#DNN_0_01_H3.txt')
fname={'DNN_0_001_H3','DNN_0_01_H3','DNN_0_1_H3','DNN_0_1_H3_2','DNN_0_1_H3'}

for i in fname:

    f=open(i+'.txt')
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

    print (tp, fp, tn, fn, tp+tn+fp+fn)
    print(accuracy, recall, precision)

    '''
    print(" TP:" + str(tp) + " FP:" + str(fp) + " TN:" + str(tn)+" FN:"+ str(fn)+ " Total :"+str(tp+tn+fp+fn))
    #9 1 8 2
    print("Accuracy : "+str(round((tp+tn)/(tp+tn+fp+fn),2)))
    print("Recall: "+str(round(tp/(tp+fp),2))) #condition positive
    print("Precision: " + str(round(tp/(tp+fn),2))) #prediction positive

    sf=open('output.txt','w')
    sf.write(" TP:" + str(tp) + " FP:" + str(fp) + " TN:" + str(tn)+" FN:"+ str(fn)+ " Total :"+str(tp+tn+fp+fn))
    sf.write(" Accuracy : "+str(round((tp+tn)/(tp+tn+fp+fn),2)))
    sf.write(" Recall: "+str(round(tp/(tp+fp),2))) #condition positive
    sf.write(" Precision: " + str(round(tp/(tp+fn),2))) #prediction positive
    sf.close()
    '''
        
    f.close()
print('finish')