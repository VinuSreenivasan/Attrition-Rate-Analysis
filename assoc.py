import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import sklearn
from matplotlib import cm
from sklearn import cluster


#%matplotlib inline

#READING THE CSV FILE AND SCATTER PLOTS
data = pd.read_csv(r'hr_data.csv')
#print data[['satisfaction_level']]
'''column_header = list(data.columns)
plt.scatter(data[[column_header[0]]], data[[column_header[10]]])
plt.show()'''

#CLUSTER

'''data_array=np.array(data[['satisfaction_level','last_evaluation']])
k=2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(data_array)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


for i in range(k):
	dsp = data_array[np.where(labels==i)]
	plt.plot(dsp[:,0],dsp[:,1],'o',markersize=7)
	lines=plt.plot(centroids[i,0],centroids[i,1],'kx')
	plt.setp(lines, ms=15.0)
	plt.setp(lines, mew=4.0)
plt.show()'''


#ASSOCIATIVE ANALYSIS

#GROUPING
data_grouping = data.groupby(['sales','left'])['sales'].count()
print data_grouping
data_sales_count = data.groupby(['sales'])['sales'].count()
data_salary_count = data.groupby(['left'])['left'].count()
#print data_salary_count
#print data_grouping

#DICTIONARY
paired_items_dict = data_grouping.to_dict()
sales_count_dict=data_sales_count.to_dict()
salary_count_dict = data_salary_count.to_dict()
#print paired_items_dict
#sprint len(sales_count_dict)

#SUPPORT
#sales_opt = dict((k,v) for k,v in sales_count_dict.items() if v >= 750)
support_pairs = dict((k,round(float(v)/len(data),3)) for k,v in paired_items_dict.items())
support_sales = dict((k,round(float(v)/len(data),3)) for k,v in sales_count_dict.items())
support_salary = dict((k,round(float(v)/len(data),3)) for k,v in salary_count_dict.items())
print support_pairs
#opt_support_pairs = dict((k,v) for k, v in support_pairs.items() if v>=0.020)
#print len(opt_support_pairs)

#CONFIDENCE
confident_pairs=dict((k,round(float(v)/support_sales[k[0]],3)) for k,v in support_pairs.items())
#print confident_pairs

#LIFT
lift_pairs = dict((k, round(float(v)/(support_sales[k[0]]*support_salary[k[1]]),3)) for k, v in support_pairs.items())
print lift_pairs


#FINAL -CONFIDENCE & LIFT
final_confidence_lift = dict((k,(confident_pairs[k],lift_pairs[k])) for k, v in support_pairs.items())
for k,v in final_confidence_lift.items():
	print k,v

color = cm.inferno_r(np.linspace(.4,.8,30))
# = [{x:random.randint(1,5)} for x in range(len(lift_pairs))]
pd.Series(lift_pairs, index = lift_pairs.keys()).plot(kind="barh", color=color)
plt.title('Associative analysis')
plt.xlabel('Lift')
plt.show()
