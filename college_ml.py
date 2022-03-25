import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import argparse


# TO-DO considerations
# 1. save model so that re-training can be skipped
# 2. read testing set from a file

def main():
#parsing input
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores',type = float)
    parser.add_argument('--input-rate', type=float, nargs=5) # --input-rate 1 2 3 4 5
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args ()


    df = pd.read_csv(r'colleges.csv')
    
    #select useful columns and replace None to 0
    df = df.iloc[:,[1,2,23,26,27,28,29,30,31,32,33,34,35]].fillna(0)
    #sat&act reading to percentage
    df = df.assign(reading_pc = ((
    df['SAT_reading_writing_upper']+df['SAT_reading_writing_lower'])/2/800+(df['ACT_english_lower']+df['ACT_english_upper'])/2/36)/2)
    #sat&act math  to percentage
    df = df.assign(math_pc = ((df['SAT_math_upper']+df['SAT_math_lower'])/2/800+(df['ACT_math_lower']+df['ACT_math_upper'])/2/36)/2)
    df = df.assign(pc = (df['reading_pc']+df['math_pc'])/2)
    #percentage of scores, and its acceptance rate
    #total_pc = df[['reading_pc','math_pc','acceptance_rate']].values
    total_pc = df[['pc','acceptance_rate']].values
#Machine Learning normalization
    X = np.array(total_pc)
    norm = MinMaxScaler().fit(X)
    X = norm.transform(X)
#Machine Learning process
    kmeans = KMeans(n_clusters=args.k, random_state=0).fit(X)
    label = kmeans.labels_
    y_km = kmeans.fit_predict(X) #clusters'labels of all insititution
    
#list of insitution name of clusters
    clusters = y_km.tolist()
    mycluster = [list () for _ in range (0, args.k+1)]   
    # index start from 1 (e.g., cluster1 == mycluster [1]) do not use mycluster [0]
    for i in range (0, len (clusters)):
        mycluster [clusters[i]].append (df.iloc [i]['institution'])
        
    
   
#process input SAT/ACT scores, convert to decimal/percentage format    
    input_scores = 0 #default 0 
    if args.scores in range(1,37):
        input_scores = args.scores/36
    elif args.scores in range(100,1601):
        input_scores = args.scores/1600
    else:
        input_scores = 0
    input_rate = args.input_rate
#ML predict and output recommend insititution 
    predict_result = kmeans.predict (
            [[input_scores, accept_rate(input_rate[0], 
             input_rate [1], input_rate[2], input_rate[3], input_rate[4])]])
    
    print(accept_rate(input_rate[0],input_rate [1], input_rate[2], input_rate[3], input_rate[4]))
    print (mycluster [int (predict_result)])

#ML Graph 
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],s=50, c='lightgreen',  marker='s', edgecolor='black', label='cluster 1' )

    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],s=50, c='orange', marker='o', edgecolor='black',label='cluster 2')

    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3'    )

    plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1],s=50, c='lightpink',marker='v', edgecolor='black',label='cluster 4'    )

    plt.scatter(X[y_km == 4, 0], X[y_km == 4, 1],s=50, c='yellow',marker='v', edgecolor='black',label='cluster 5'    )   

# plot the centroids
    plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids')


    plt.legend(scatterpoints=1)
    plt.grid()
#uncomment the next line to show the graph
    #plt.show()

#acceptance rate calculation 
def accept_rate(gpa,awards,leadership,scholarship,ap):
    gpa = min (4.0, max (0.0, gpa))
    df = pd.read_csv(r'colleges.csv')
    school_arate = df.loc[:,['acceptance_rate']].fillna(0)           
    rate = 20
    #pc:percentage 
    #assume the full score of gpa=4,awards=1,leadership=3,scholarsp=3,ap=5)
    gpapc = gpa/4*rate
   
    if awards >= 1:
        awardspc = rate
    else:   
        awardspc = 0
    if leadership >= 3:
        leadershippc = rate
    else:
        leadershippc = leadership /3 *rate
    if scholarship >= 3:
        scholarshippc = rate
    else:    
        scholarshippc = scholarship /3 *rate
    if ap >=5:
        appc = rate
    else:
        appc = ap /5*rate
    
    accp_rate = gpapc+awardspc+leadershippc+scholarshippc+appc
    accp_rate = 100 - accp_rate
    return accp_rate*0.01
                                                                                        
if __name__ == '__main__':
    main()
     
