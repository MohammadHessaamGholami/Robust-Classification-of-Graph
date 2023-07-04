from hashlib import new
from os import WEXITED
from networkx.classes.function import number_of_edges
import numpy as np
import networkx as nx
from numpy.core.numeric import identity
import pylab as plt
from sklearn.model_selection import train_test_split
import random
import sys
######## get dataset from networkx library
name_of_dataset = sys.argv[1]
percentage_list = sys.argv[2].split(',')
percentage_list = [int(x) for x in percentage_list]
coeff_list = sys.argv[3].split(',')
coeff_list = [float(x) for x in coeff_list]
iteration = int(sys.argv[4])
#print(percentage_list)
#print(coeff_list)

if name_of_dataset == "polblogs":
    print("dataset polblogs working ....")
    f = open("polblogs/adjacency.csv", "r")
    h = open("polblogs/labels.csv", "r")
    label = np.array([])
    for index,line in enumerate(h):        
        label = np.append(label, int(line))
    h.close()
    number_of_nodes = len(label)
    number_of_edges = 0
    for index,line in enumerate(f):
        number_of_edges += 1
        info = line.split('\n')[0]
        info = info.split(', ')
        if(index == 0):
            wieghts_matrix = np.zeros((number_of_nodes, number_of_nodes))
            continue
        node_1 = int(info[0])
        node_2 = int(info[1])
        wieghts_matrix[node_1-1][node_2-1] = 1
        wieghts_matrix[node_2-1][node_1-1] = 1
    f.close()
    print(number_of_edges)

    degrees = np.sum(wieghts_matrix, axis=1)
    m = len(wieghts_matrix)
    i = 0
    c = 0
    print(np.shape(degrees))
    while i < m:
        if degrees[i] == 0:
            m = m-1
            c += 1
            degrees = np.delete(degrees, i)
            wieghts_matrix = np.delete(wieghts_matrix, i, axis=0)
            wieghts_matrix = np.delete(wieghts_matrix, i, axis=1)
            number_of_nodes = number_of_nodes -1
            label = np.delete(label, i)
            i = i-1
        i += 1
elif name_of_dataset == "karate" :
    print("dataset karate working ....")
    kc_graph = nx.karate_club_graph()
    number_of_nodes = kc_graph.number_of_nodes()
    number_of_edges = kc_graph.number_of_edges()
    wieghts_matrix = nx.to_numpy_array(kc_graph)
    label = np.asarray(
            [kc_graph.nodes[i]['club'] != 'Mr. Hi' for i in kc_graph.nodes]).astype(np.int64)
else:
    print("can't understand: dataset karate working ....")
    kc_graph = nx.karate_club_graph()
    number_of_nodes = kc_graph.number_of_nodes()
    number_of_edges = kc_graph.number_of_edges()
    wieghts_matrix = nx.to_numpy_array(kc_graph)
    label = np.asarray(
            [kc_graph.nodes[i]['club'] != 'Mr. Hi' for i in kc_graph.nodes]).astype(np.int64)


new_label = [label[i] if label[i]==1 else -1 for i in range(len(label))]
#percentage_list = [1,2,5,10,20,50]  
#coeff_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#percentage_list = [50]
#coeff_list = [0.7]
best_accuracy_list = []
best_coeff_list = []
for per in percentage_list:
    best_accuracy = 0
    best_coeff = 0
    for coef in coeff_list:
        accuracy_total = 0
        for iter in range(iteration):
            ######## choose labeled data based on percentage value
            X = np.arange(number_of_nodes)
            percentage = per
            k = len(X) * percentage // 100
            with_label_indicies = random.sample(list(X), k)
            without_label_indices = set(X) - set(with_label_indicies)
            without_label_indices = np.array(list(without_label_indices))
            y = [new_label[i] if i in with_label_indicies else 0 for i in range(number_of_nodes)]

            ######## create D matrix and after that calculate LN matrix
            D_matrix = np.zeros((number_of_nodes, number_of_nodes))
            D_half = np.zeros((number_of_nodes, number_of_nodes))
            degrees = np.sum(wieghts_matrix, axis=1)


            for i in range(len(wieghts_matrix)):
                D_matrix[i][i] = degrees[i]
                D_half[i][i] = (degrees[i])**(0.5)

            D_half = np.divide(1.0, D_half, out=np.zeros_like(D_half), where=D_half!=0 )
            L = D_matrix-wieghts_matrix
            x_1 = np.matmul(D_half, L )
            LN = np.matmul(x_1, D_half)
            #print(LN)

            ######## calculate eigenvalue and eigenvector of LN matrix
            eval_LN, evec_LN = np.linalg.eig(LN)

            idx = eval_LN.argsort()[::-1]   
            eigenValues = eval_LN[idx]
            V_matrix = evec_LN[:,idx]

            eigenValues = np.flip(eigenValues) #smallest eigenValue is eigenValues[0]
            V_matrix = np.flip(V_matrix, axis=1) #V_matrix[:,0] is associated to smallest eigenValue


            ######## calculate v0 vector
            v0 = np.sqrt(degrees)
            v0 = v0/np.linalg.norm(v0)
            v0 = np.reshape(v0, (number_of_nodes,1))

            ######## calculate gama
            #coef = 0.9
            if eigenValues[1] != 0:
                #print("*************************")
                gama = coef * eigenValues[1]
            else:
                x_1 = np.matmul(D_half, wieghts_matrix)
                S_matrix = np.matmul(x_1, D_half)
                eval_LN, evec_LN = np.linalg.eig(S_matrix - np.matmul(v0, v0.T))

                idx = eval_LN.argsort()[::-1]   
                eigenValues = eval_LN[idx]
                V_matrix = evec_LN[:,idx]
                #print(eigenValues)
                gama = coef * eigenValues[0]
                #gama = eigenValues[0]

            ######## calculate final f 
            p1 = np.divide(LN, gama)
            p1 = p1 - identity(number_of_nodes)
            #p2 = np.divide(1.0, p1, out= np.zeros_like(p1), where=p1!=0)
            #print(p1)
            #p2 = np.linalg.inv(p1)
            p2 = np.linalg.pinv(p1)

            y = np.reshape(y, (number_of_nodes,1))
            p3 = y - np.matmul(np.matmul(v0, v0.T), y)
            p3 = p3.reshape(number_of_nodes, 1)
            f = np.matmul(p2 , p3)
            output = np.sign(f)

            ######## calculate accuracy of algorithm
            count = 0
            for index in without_label_indices:
                if output[index] == new_label[index]:
                    count += 1
            accuracy = count/len(without_label_indices)
            accuracy_total += accuracy
            print("(percentage: %d, coef: %f, iteration: %d)accuracy of algorithm is %f%%"%(per,coef,iter,accuracy*100))
            #print(f)
        acc = 100*accuracy_total/iteration
        if acc > best_accuracy:
            best_accuracy = acc
            best_coeff = coef
        #print("(%d%% of data have label)average accuracy of algorithm is %f%%"%(per, 100*accuracy_total/iteration))
    best_accuracy_list.append(best_accuracy)
    best_coeff_list.append(best_coeff)
print("best accuracy list for different percentage:", best_accuracy_list)
plt.figure()
plt.xlabel("percentage of labeled data(%)")
plt.ylabel("accuracy of algorithm(%)")
plt.plot(percentage_list, best_accuracy_list, marker='*')
plt.savefig("RobustGC_Accuracy_%s_%d"%(name_of_dataset, iteration))

plt.figure()
plt.xlabel("percentage of labeled data(%)")
plt.ylabel("best coef")
plt.plot(percentage_list, best_coeff_list, marker='*')
plt.savefig("RobustGC_coefficient_%s_%d"%(name_of_dataset, iteration))

plt.show()





"""

'''
f = open("soc-karate.mtx", "r")
for index,line in enumerate(f):
    info = line.split()
    if(index == 0):
        number_of_nodes = int(info[0])
        number_of_edges = int(info[2])
        wieghts_matrix = np.zeros((number_of_nodes, number_of_nodes))
        continue
    node_1 = int(info[0])
    node_2 = int(info[1])
    wieghts_matrix[node_1-1][node_2-1] = 1
    wieghts_matrix[node_2-1][node_1-1] = 1
'''


#print(label)
def get_y(number_of_nodes, eigenVector):
    coefficient = np.zeros((number_of_nodes, number_of_nodes))
    for eq in range(number_of_nodes):
        for y in range(number_of_nodes):
            for j in range(number_of_nodes):
                coefficient[eq][y] += (eigenVector[eq][j]*eigenVector[y][j])
    for i in range(number_of_nodes):
        coefficient[i][i] = coefficient[i][i] -1
    #coefficient = coefficient * (10**15)         
    #print("**")
    #print(coefficient)
    y = np.linalg.solve(coefficient, np.ones(number_of_nodes))
    return y
#print(np.matmul(LN, evec_LN[:,0].T))
#print(eval_LN[0]*evec_LN[:,0].T)
#print("**///////////////////////////////////////////")


#print(np.matmul(LN, V_matrix[:,0]))
#print(eigenValues[0]*V_matrix[:,0])
#print("*************************")
#
# print(get_y(number_of_nodes, V_matrix))

    #print(node_1, node_2)
#print(number_of_nodes)
#print(number_of_edges)
#print(len(wieghts_matrix[0]))

# print(index)
#   print(line)
#print(f.read())

#x_1 = np.matmul(D_half, wieghts_matrix)
#S_matrix = np.matmul(x_1, D_half)
#LN = I - S_matrix

#print(f)
#y = get_y(number_of_nodes, V_matrix)
"""