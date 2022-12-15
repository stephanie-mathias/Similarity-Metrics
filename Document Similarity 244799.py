#!/usr/bin/env python
# coding: utf-8

# # Comparative Analyses of Corpus Similarity Methods in Python

# <b>Candidate: 244799</b><br/>
# Word Count (including headings, excluding references, code and output): 1975

# Methods for computing the similarity of bodies of texts, also known as corpuses, has many applications in academic and commercial settings, for example finding similar webpages or identifying plagiarism. Two ways of finding corpus similarity are the Jaccard and Cosine similarity measures. These will both be described theoretically then implemented in Python and tested on a range or corpuses to show their efficiency. Finally, a form of parallel computing is tested for further optimisation for similarity computation. The aim is to provide an overview of the most favourable similarity methods on texts and their limitations.

# In[1]:


#import Python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing
from multiprocessing import Pool
import random
import time
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from random import sample
import scipy.stats as stats
import pandas as pd
from collections import defaultdict

#set seed so random variables are repeatable
random.seed(47)


# A function is created to generate texts which takes two inputs; <i>m</i> the number of random documents to be generated and <i>n</i>, the number of words within each document. Words are pulled from a pool of 5000 random words, <i>n</i> number of times and contained within a string.<br/>

# In[2]:


#create 5000 random words
random_words = word_tokenize(' '.join(sample(words.words(), 5000)))

#function: creates m number of documents with n number of random words from the random_words list
def rand_docs(m,n):
    texts = []
    while len(texts) < m:
        texts.append(" ".join(random.choices(random_words,k=n)))
    return texts


# In this context, similarity of texts is defined as the number of identical  words that appear in both texts and their frequencies. It is acknowledged that document similarity can also include other attributes, such as sentence length or word proximity, but these will not contribute to any similarity metric in this analysis.

# ## Jaccard Similarity

# The Jaccard similarity measure is a ratio of the size of the intersection of two sets of data to the size of the union, the total count of elements of the sets<sup>1</sup>. It is described by the following equation for sets <i>A</i> and <i>B</i>:<br/>
# $Jacc(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$
# <br/><br/>

# In Python, strings are converted into bags of words represented in dictionaries, where each key is a unique word in the text and the value its frequency. The intersection size is found by summing all minimum values for a matching keys in both bags. The union size is the total values of both bags subtracted by the intersection value.

# ### Jaccard Similarity: Theoretical Running Time

# The most time-intensive step in the Jaccard similarity algorithm on two bags of words is finding the intersection size, where the keys of one bag must be iterated and the corresponding key, if it exists, found in the other bag. Finding keys runs at constant time due to the hash-function for dictionaries implemented in Python<sup>2</sup>. Therefore, the run-time should increase with the increasing length of one bag, a run time of $O(n)$.<br/>

# ### Jaccard Similarity: Impirical Running Time

# Running time is averaged for the Jaccard function applied to varying lengths of documents represented as bags of words. The method is optimised by selecting the shortest length bag to iterate through. This assumes the extra step of comparing bag lengths is quicker than picking a bag at random to iterate through, which may be much longer than its corresponding bag.
# Below are functions to convert documents stored as strings to bags of words, a function to total dictionary values and the Jaccard function taking a list of two bags of words as input.

# In[3]:


#function: tokenises lists of words
def token_texts(texts):
    bags = []
    for text in texts:
        bags.append(word_tokenize(text))
    return bags

#function: transforms list of tokens into dictionary of word counts
def dict_bags(tokenslist):
    bag_dicts = []
    for tokens in tokenslist:
        bag_dict = {}
        for token in tokens:
            bag_dict[token] = bag_dict.get(token,0)+1
        bag_dicts.append(bag_dict)
    return bag_dicts

#function: gets total of values in a dictionary
def dict_total(dict1):
    values = dict1.values()
    return sum(values)

#function: times the run time of a function
def time_it(func,*args,repeats=5,**kwargs):
    times = []
  
    while repeats > 0:
        start = time.time()
        ans = func(*args,**kwargs)
        end = time.time()
        total_time = end-start
        times.append(total_time)
        repeats -=1
    
    mean = np.mean(times)
    std = np.std(times)
    error = std/(len(times)**0.5)
 
    return (mean,error,std)

#function: compute jaccard similarity of two bags of tokens
def jaccard(bags):
    if len(bags) != 2:
        print('Input must be two bags only')
    else:
        
        #find which of the two bags dictionaries is the shortest
        mindict = None 
        maxdict = None
        mindictlen = min(len(bags[0]),len(bags[1]))
        
        if len(bags[0]) == mindictlen:
            mindict = bags[0]
            maxdict = bags[1]
        else:
            mindict = bags[1]
            maxdict = bags[0]
        
        #calculate the similarity
        intersection = {}
        for item in mindict.keys():
            if item in maxdict.keys():
                intersection[item] = min(mindict[item],maxdict[item])
        
        intersection_value = dict_total(intersection)
        union_value = dict_total(bags[0])+dict_total(bags[1])-intersection_value
        jaccard = intersection_value / union_value
        return jaccard, mindictlen


# The Jaccard measure is computed for documents 100 to 20,000 words, at intervals of 100.

# In[4]:


#Run jaccard similarity on increasing size texts
doclengths = [x for x in range(100,20001,100)]
x_baglength = []
y_avruntime = []
y_errorruntime = []

for length in doclengths:
    docs = rand_docs(2,length)
    tokens = token_texts(docs)
    bags = dict_bags(tokens)
    bag_len = jaccard(bags)[1]
    
    mean,error,std = time_it(jaccard,bags)
    x_baglength.append(bag_len)
    y_avruntime.append(mean)
    y_errorruntime.append(error)


# The average and logarithmic runtime against length of the shortest dictionary are plotted.

# In[5]:


#Plot the outcomes
plt.subplots(1,2, figsize=(14,6))
plt.suptitle("Average Run Time: Jaccard Similarity", fontsize=20)

#Plot raw data
plt.subplot(1,2,1)
plt.plot(x_baglength,y_avruntime,c='cadetblue',linewidth=3.0)
plt.xlabel("Bag Length")
plt.ylabel("Mean Time (Seconds)")

#Calculate and plot log10 of raw data
plt.subplot(1,2,2)
log_baglen=np.log10(x_baglength)
log_runtimejac=np.log10(y_avruntime)

plt.plot(log_baglen,log_runtimejac,c='darkgrey',linewidth=3.0)
plt.xlabel("log10 Bag Length")
plt.ylabel("Log10 Mean Time (Seconds)")

plt.tight_layout()
plt.show()


# In[6]:


#perform linear regression to get the gradient and intercept
slope,intercept,r_value,p_value,std_err = stats.linregress(log_baglen,log_runtimejac)
print(f"The gradient is:\n {round(slope,3)} ")
interceptorg = 10**intercept
print(f"The intercept is:\n {interceptorg}")


# The gradient of the logarithmic graph is approximately 1.4, indicating a run time of O(n<sup>1.4</sup>), slightly longer than the $O(n)$ proposed, with a constant of 1.5x10<sup>-8</sup>. A possible explanation is the extra time required to sum the values in both dictionaries.

# ## Cosine Similarity

# An alternative method is to represent bags of words as vectors, and then use the cosine formula for calculating the cosine angle between the vectors which is the product of the two vectors divided by the product of their magnitudes<sup>3</sup>. Smaller cosine angles indicate greater similarity of the vectors and therefore greater document similarity. The cosine similarity is described below:<br/>
# $
# \begin{align}
# \cos(\theta )={\mathbf {A} \cdot \mathbf {B} \over \|\mathbf {A} \|\|\mathbf {B} \|}
# \end{align}
# $

# ### Cosine Similarity: Theoretical Running Time

# The theoretical runtime for cosine similarity method applied to two vectors is $O(n)$, as the most time-intensive step is iterating through the index range of the length of the vectors to calculate the total dot product which will increase at longer vector lengths.

# ### Cosine Similarity: Empirical Running Time

# A function is made to convert the bags of words to dense vectors. The cosine function created then takes these as input. Since the dot product of vectors can be obtained through element-wise multiplication and addition or by using the numpy library dot product function, two functions will be compared.

# In[7]:


#function that converts two bags of words into dense vectors
def bags_to_dense_vectors(bags):
    if len(bags) != 2:
        print('Input must be two bags only')
        
    else:
        dict1 = bags[0]
        dict2 = bags[1]
        
        for key in dict1.keys():
            if key not in dict2.keys():
                dict2[key] = 0
        for key2 in dict2.keys():
            if key2 not in dict1.keys():
                dict1[key2] = 0
        
        vector1 = []
        vector2 = []
        for key2, value2 in dict1.items():
            vector1.append(value2)
            vector2.append(dict2[key2])
        
        return [vector1, vector2]
        
#function: find the dot product of two vectors/lists
def dot(v1,v2):
    total = 0
    for i in range(0,len(v1)):
        total += v1[i]*v2[i]
    return total

#function: computes cosine similarity on two vectors using multiplication and summation
def cosine_sim(vectors):
        cosinesim = dot(vectors[0],vectors[1])/(dot(vectors[0],vectors[0])*dot(vectors[1],vectors[1]))**0.5
        return cosinesim, len(vectors[0])
    
#function: run cosine similarity using numpy library
def cosine_sim2(vectors):        
        cosinesim = np.dot(vectors[0],vectors[1])/(np.dot(vectors[0],vectors[0])*np.dot(vectors[1],vectors[1]))**0.5
        return cosinesim, len(vectors[0])


# The run times of these methods are tested on same range of corpuses as before.

# In[8]:


#Run cosine similarity (both types) on increasing size texts
x_vectorlength = []
y_avruntimecos = []
y_errorruntimecos = []
y_avruntimecos_np = []
y_errorruntimecos_np = []

for length in doclengths:
    docs = rand_docs(2,length)
    tokens = token_texts(docs)
    bags = dict_bags(tokens)
    vectors = bags_to_dense_vectors(bags)
    vectorlen = len(vectors[0])
    x_vectorlength.append(vectorlen)
    
    mean,error,std = time_it(cosine_sim,vectors)
    y_avruntimecos.append(mean)
    y_errorruntimecos.append(error)
    
    mean2,error2,std2 = time_it(cosine_sim2,vectors)
    y_avruntimecos_np.append(mean2)
    y_errorruntimecos_np.append(error2)


# In[9]:


#Plot run times of cosine similarity 
plt.subplots(1,2, figsize=(14,6))
plt.suptitle("Average Run Time: Cosine Similarity on Vectors", fontsize=17)

#Plot raw data
plt.subplot(1,2,1)
y1_ledg = mpatches.Patch(color='cornflowerblue', label='cosine')
y2_ledg = mpatches.Patch(color='lightsteelblue', label='cosine numpy')
plt.plot(x_vectorlength,y_avruntimecos,c='cornflowerblue',linewidth=3.0)
plt.plot(x_vectorlength,y_avruntimecos_np,c='lightsteelblue',linewidth=3.0)
plt.legend(handles=[y1_ledg,y2_ledg])
plt.xlabel("Dense Dictionary Length",fontsize=12)
plt.ylabel("Mean Time (Seconds)",fontsize=12)

#Calculate and plot log10 of raw data
plt.subplot(1,2,2)
log_vectorlen=np.log10(x_vectorlength)
log_runtime=np.log10(y_avruntimecos)
log_runtime_np=np.log10(y_avruntimecos_np)

plt.plot(log_vectorlen,log_runtime,c='cornflowerblue',linewidth=3.0)
plt.plot(log_vectorlen,log_runtime_np,c='lightsteelblue',linewidth=3.0)
plt.legend(handles=[y1_ledg,y2_ledg])
plt.xlabel("log10 of Dense Dictionary Length",fontsize=12)
plt.ylabel("Log10 Mean Time (Seconds)",fontsize=12)

plt.tight_layout()
plt.show()


# In[10]:


#perform linear regression to find the gradient and intercept
slope2,intercept2,r_value2,p_value2,std_err2 = stats.linregress(log_vectorlen,log_runtime)
slope3,intercept3,r_value3,p_value3,std_err3 = stats.linregress(log_vectorlen,log_runtime_np)
interceptorg2 = 10**intercept2
interceptorg3 = 10**intercept3
print(f"The gradient for naive cosine similarity is: {round(slope2,3)} \n and the intercept is: {interceptorg2}")
print(f"The gradient for cosine similarity using numpy is: {round(slope3,3)} \n and the intercept is: {interceptorg3}")


# The gradient for the first cosine is approximately 1, consistent with the prediction of $O(n)$ with a constant of 3 x 10<sup>-7</sup>. The numpy form was quicker, with a gradient of 0.81, but had a larger constant of 2 x 10<sup>-6</sup>. It has been documented that the dot product of arrays of numbers can be calculated up to 70 times faster than a manual loop method in Python<sup>4</sup>, however underlying this, numpy restructures the data which may explain the larger constant.

# ## Cosine Similarity: Alternative Method

# Another way of computing the cosine similarity without using vectors is creating a function which takes bags of words and loops over them to compute the dot products. It also sums up the magnitudes of each bag by summing the square of all of the values.

# In[11]:


#function: calculate magnitude of dictionary values
def mag_dict_values(v):
    total = 0
    for i in v.values():
        total += i**2
    return total

#function: calculate cosine without using vectors
def cosine_dicts(dicts):
    dotprod = 0
    for key, value in dicts[0].items():
        if key in dicts[1].keys():
            dotprod += dicts[0][key]*dicts[1][key]
            
    cosine = dotprod / (mag_dict_values(dicts[0]) * mag_dict_values(dicts[1]))**0.5
    return cosine


# ### Cosine Similarity Alternative Method: Theoretical Running Time

# The theoretical runtime of this approach is $O(n)$, plus some constant due to the iteration over the full length of one dictionary, so that it can find the corresponding key in another dictonary.

# ### Cosine Similarity Alternative Method: Empirical Running Time

# This method is timed on the same range of increasing texts as before, and the mean times and their logarithms plotted.

# In[12]:


#Run dictonary cosine on increasing size texts
x_cosdictlength = []
y_avruntime_cd = []
y_errorruntime_cd = []

for length in doclengths:
    docs = rand_docs(2,length)
    tokens = token_texts(docs)
    bags = dict_bags(tokens)
    dictlength = len(bags[0])
    
    mean,error,std = time_it(cosine_dicts,bags)
    x_cosdictlength.append(dictlength)
    y_avruntime_cd.append(mean)
    y_errorruntime_cd.append(error)


# In[13]:


#Plot run times of cosine similarity on dictionaries
plt.subplots(1,2, figsize=(14,6))
plt.suptitle("Average Run Time: Cosine Similarity on Dictionaries", fontsize=17)

#Plot raw data
plt.subplot(1,2,1)
plt.plot(x_cosdictlength,y_avruntime_cd,c='skyblue',linewidth=3.0)
plt.xlabel("Sparse Dictionary Length",fontsize=12)
plt.ylabel("Mean Time (Seconds)",fontsize=12)

#Calculate and plot log10 of raw data
plt.subplot(1,2,2)
log_dictlen=np.log10(x_cosdictlength)
log_runtime_cd=np.log10(y_avruntime_cd)

plt.plot(log_dictlen,log_runtime_cd,c='slategrey',linewidth=3.0)
plt.xlabel("log10 of Sparse Dictionary Length",fontsize=12)
plt.ylabel("Log10 Mean Time (Seconds)",fontsize=12)

plt.tight_layout()
plt.show()


# In[14]:


#perform linear regression to find the slope and intercept
slope4,intercept4,r_value4,p_value4,std_err4 = stats.linregress(log_dictlen,log_runtime_cd)
print(f"The gradient is:\n {round(slope4,3)}")
interceptorg4 = 10**intercept4
print(f"The intercept is:\n {interceptorg4}")


# The gradient is approximately 1, suggesting that the $O(n)$ is correct. It also has a constant of approximately 6 x 10<sup>-7</sup>.

# ### Cosine Similarity Alternative Method: Correctness

# This method is tested for consistency with the cosine method taking dense vectors as input by applying them to two texts of 1000 random words at a time, 100 times over. A score of 1 is given if the similarity values match and 0 if not, then an overall percentage of consistency is calculated.

# In[15]:


#get 100 pairs of texts
texts = []
while len(texts) < 100:
    textduo = rand_docs(2,1000)
    texts.append(textduo)
   
sim_values = []
matches = []

#checks if the outputs of the two functions are the same
#if they are, '1' is added to the matches list, if they aren't '0' is added, this is used to calculate the overall percentage
for duo in texts:
    tokens = token_texts(duo)
    bags = dict_bags(tokens)
    vectors = bags_to_dense_vectors(bags)

    sim1 = cosine_sim(vectors)
    sim2 = cosine_dicts(bags)
    sim_values.append((sim1[0],sim2))
    if sim1[0] == sim2:
        matches.append(1)
    else:
        matches.append(0)

#calculate the percentage of measures that were equal        
percentage_match = sum(matches)/len(matches)*100
print(f"The percentage of matching values is: {int(percentage_match)} %")


# There is 100% match so the methods are consistent.

# ### Similarity Methods Comparison

# The average run time orders and constants for all methods tested are plotted on one graph and tabulated.

# In[16]:


#create run times figures
plt.subplots(1,2, figsize=(14,6))
plt.suptitle("Average Run Time: All Measures Comparison", fontsize=17)

#plot data
plt.subplot(1,2,1)
ledg1 = mpatches.Patch(color='lightblue', label='jaccard')
ledg2 = mpatches.Patch(color='cornflowerblue', label='cosine')
ledg3 = mpatches.Patch(color='mediumseagreen', label='cosine numpy')
ledg4 = mpatches.Patch(color='tomato', label='cosine dictionaries')

plt.scatter(x_baglength,y_avruntime,c='lightblue')
plt.scatter(x_vectorlength,y_avruntimecos,c='cornflowerblue')
plt.scatter(x_vectorlength,y_avruntimecos_np,c='mediumseagreen')
plt.scatter(x_cosdictlength,y_avruntime_cd,c='tomato')
plt.legend(handles=[ledg1,ledg2,ledg3,ledg4])
plt.xlabel("Input Length",fontsize=12)
plt.ylabel("Mean Time (Seconds)",fontsize=12)

#plot log values
plt.subplot(1,2,2)

plt.scatter(log_baglen,log_runtimejac,c='lightblue')
plt.scatter(log_vectorlen,log_runtime,c='cornflowerblue')
plt.scatter(log_vectorlen,log_runtime_np,c='mediumseagreen')
plt.scatter(log_dictlen,log_runtime_cd,c='tomato')
plt.legend(handles=[ledg1,ledg2,ledg3,ledg4])
plt.xlabel("log10 of Input Length",fontsize=12)
plt.ylabel("Log10 Mean Time (Seconds)",fontsize=12)

plt.tight_layout()
plt.show()

#create table of impirical run times and constants
fig, ax = plt.subplots(1,1,figsize=(16,2))
plt.figure(facecolor='skyblue')
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

all_data={'Jaccard':{'Impirical RunTime':round(slope,3),'Constant':interceptorg},
          'Cosine':{'Impirical RunTime':round(slope2,3),'Constant':interceptorg2},
          'Cosine numpy':{'Impirical RunTime':round(slope3,3),'Constant':interceptorg3},
          'Cosine Dictionaries':{'Impirical RunTime':round(slope4,3),'Constant':interceptorg4}}

all_values = pd.DataFrame(data=all_data)
table_colours = [["lightblue", "cornflowerblue", "mediumseagreen", "tomato"],
                 ["lightblue", "cornflowerblue", "mediumseagreen", "tomato"]]

table = ax.table(cellText=all_values.values,
                 rowLabels=all_values.index,
                 cellColours = table_colours,
                 colLabels=all_values.columns,
                 cellLoc='right',
                 loc = "center")
ax.set_title('Impirical Runtime and Constant Table',fontsize=40)

table.auto_set_font_size(False)
table.set_fontsize(25)
table.scale(1.7, 2)

plt.show()


# The cosine from vectors, particularly using numpy, produce quicker running orders than the cosine dictionary and Jaccard similarity methods. Conversely, the Jaccard has the smallest constant and therefore at small document size may be quicker than other methods. Meanwhile, the numpy cosine has the larger constant and may take more time for smaller document sizes.

# ## All-Pairs Similarity of Multiple Documents 

# A function is created combining these methods and applying them to a group of documents for all pair-wise similarity. The first argument is a list of documents and the second argument takes the method to be executed. The function indexes the documents and gets the list of pairs of unique indices. The output is a table with the unique pair indexes as the first column and the similarity value in the second.
# <br/><br/>

# In[17]:


#function: takes as input a similarity measure and a number of documents represented in dictonaries
def sim_documents(docslist,measure):
    
    #creates a dictionary to label each document with a number
    docsdict = {}
    docs = len(docslist)
    index = 1
    for doc in docslist:
        docsdict[index] = doc
        index += 1
        
    #get unique pairs of documents, excluding a document paired with itself
    pairs = []
    for i in docsdict.keys():
        for j in docsdict.keys():
            if i != j:
                pair = sorted([i,j])
                pairs.append(pair)
    
    uniquepairs = []
    for p in pairs:
        if p not in uniquepairs:
            uniquepairs.append(p)
       
    pairs_sim_measures = {}
    
    #jaccard
    if measure == 'jaccard':
          for p1 in uniquepairs:
                docs2 = [docsdict[p1[0]],docsdict[p1[1]]]
                sim_value = jaccard(docs2)
                keyname = str(p1[0])+', '+str(p1[1])
                pairs_sim_measures[keyname] = sim_value[0]
        
    #naive cosine   
    elif measure == 'cosine':
        for p1 in uniquepairs:
            docs2 = [docsdict[p1[0]],docsdict[p1[1]]]
            sparsedocs = bags_to_dense_vectors(docs2)
            keyname = str(p1[0])+', '+str(p1[1])
            sim_value = cosine_sim(sparsedocs)
            pairs_sim_measures[keyname] = sim_value[0]
        
    #cosine with numpy    
    elif measure == 'cosinenp':
        for p1 in uniquepairs:
            docs2 = [docsdict[p1[0]],docsdict[p1[1]]]
            sparsedocs = bags_to_dense_vectors(docs2)
            keyname = str(p1[0])+', '+str(p1[1])
            sim_value = cosine_sim2(sparsedocs)
            pairs_sim_measures[keyname] = sim_value[0]
    
    #cosine through dictionaries
    elif measure == 'cosinedict':
        for p1 in uniquepairs:
            docs2 = [docsdict[p1[0]],docsdict[p1[1]]]
            keyname = str(p1[0])+', '+str(p1[1])
            sim_value = cosine_dicts(docs2)
            pairs_sim_measures[keyname] = sim_value
        
    else:
        print('Invalid measure input.')
        
    cols={0:'pairs',1:measure+' similarity'}
    pairs_sim_df = pd.DataFrame(pairs_sim_measures.items()).rename(columns=cols)   
    return pairs_sim_df


# ### All-Pairs Similarity: Theoretical Running Time

# To estimate theoretical running time for similarity of one of the methods on a list of documents size (<i>m</i>), the number of comparisons to take place has to be calculated. For any list of documents, document <i>i</i> in list of <i>m</i> documents cannot be compared to itself and since the similarity value for n[i] and n[j] is the same as n[j] to n[i], duplicate comparisons are removed.
# 
# The number of combinations (C) for number of documents (m) will be:<br/>
# $C=\frac{m(m-1)}{2}$
# <br/><br/>
# Since all runtimes are order $O(n)$, for a pair-wise similarity for <i>m</i> number of documents with <i>n</i> number of words produces the equation: <br/><br/>
# $O(\frac{m(m-1)}{2} n)$
# <br/><br/>
# The cosine methods which take vectors as input will take extra time, since these require the additonal step of converting documents into vectors.
# 

# ### All-Pairs Similarity Example: 200K Documents

# For 200,000 documents there would be 1.99999x10<sup>10</sup> unique combinations. All order run times are roughly $O(n)$ so the theoretical run time would 1.99x10<sup>10</sup> multiplied by the average length of all the documents. If the average length of the documents is 1000 words, then the estimated time would be 1.99x10^8 seconds, or just over 2303 days.

# ## All-Pairs Similarity with Parallel Computing

# Parallel computing methods allow execution of computations on subsets of data synonymously by distributing them across different processing units.<sup>5</sup> This is often applied commercially through data centres or can be set up on personal computers, where tasks are distributed across processing cores of the central processing unit.

# A form of parallel computing is implemented with the cosine numpy function on a list of documents. The functions will map the pairs to different cores, where the function is executed on each pair at the particular core it is mapped to. This uses the multiprocessing package and separate Python file (cosine_sim.py).

# In[18]:


#function processes list of documents into sparse vectors for the cosine similarity
def cosine_pre_processing(docslist):
    
    #creates a dictionary to label each document with a number
    docsdict = {}
    docs = len(docslist)
    index = 1
    for doc in docslist:
        docsdict[index] = doc
        index += 1
        
    #get unique pairs of documents, excluding a document paired with itself
    pairs = []
    for i in docsdict.keys():
        for j in docsdict.keys():
            if i != j:
                pair = sorted([i,j])
                pairs.append(pair)
    
    uniquepairs = []
    for p in pairs:
        if p not in uniquepairs:
            uniquepairs.append(p)
    
    docsoutput = []
    #naive cosine   
    for p1 in uniquepairs:
        docs2 = [docsdict[p1[0]],docsdict[p1[1]]]
        sparsedocs = bags_to_dense_vectors(docs2)
        keyname = str(p1[0])+', '+str(p1[1])
        docsoutput.append([keyname,sparsedocs])
        
    return docsoutput


# In[19]:


import cosine_sim 

#function: maps function to different cores
def mapping_processing(documents,core):
    pool = multiprocessing.Pool(processes=core)
    result = pool.map(cosine_sim.cosine_sim2,documents)
    
    cols2={0:'pairs',1:'cosine similarity'}
    result_df = pd.DataFrame(result).rename(columns=cols2)   
    return result_df


# ### All-Pairs Similarity with Parallel Computing: Efficiency Comparison

# The parallel method of calculating cosine is compared with non-paralleled computation for 10 to 30 documents with increasing intervals of 5.

# In[20]:


#create range of document amounts and for similarity
x_docs_efcom = [x for x in range(10,31,5)]
y_parallel_running = []
y_normal_running = []

for efcom in x_docs_efcom:
    testdocs_10 = rand_docs(efcom,1000)
    tokensdocs_10 = token_texts(testdocs_10)
    dict_tokensdocs_10 = dict_bags(tokensdocs_10)
    processed_docs_10 = cosine_pre_processing(dict_tokensdocs_10)
    
    mean1,std1,error1 = time_it(mapping_processing,processed_docs_10,2)
    y_parallel_running.append(mean1)
    mean2,std2,error2 = time_it(sim_documents,dict_tokensdocs_10,'cosinenp')
    y_normal_running.append(mean2)


# In[22]:


#Plot data
y1_ledgcomp = mpatches.Patch(color='xkcd:sky', label='parallel computing')
y2_ledgcomp = mpatches.Patch(color='xkcd:turquoise blue', label='regular computing')
plt.plot(x_docs_efcom,y_parallel_running,c='xkcd:sky',linewidth=3.0)
plt.plot(x_docs_efcom,y_normal_running,c='xkcd:turquoise blue',linewidth=3.0)
plt.legend(handles=[y1_ledgcomp,y2_ledgcomp])
plt.xlabel("Number of Documents",fontsize=12)
plt.ylabel("Mean Time (Seconds)",fontsize=12)
plt.title("Parallel vs. Non-Parallel Computing Running Time for Cosine Pairs-wise Similarity of Documents",size=8)

plt.tight_layout()
plt.show()


# For fewer documents (10), the regular method for computing the cosine is more efficient, alluding to a higher constant. This may be due to the extra steps required for its execution, such as running the multiprocessing library or opening the separate python file, totalling more time than the reduction when computing in parallel. However, at increasing numbers of documents (>15), the parallel computing methods far outperforms the original method.

# ### All-Pairs Similarity with Parallel Computing: Correctness

# To validate the parallel computing function, outputs from this and that of the non-parallel method are compared. Both methods will be run for 5 sets of 10 documents of 100 words. The percentage of consistent similarity values is then generated.

# In[23]:


#variables to count the number of inconsitent matches and rounds
no_match_count = []
ptest_rounds = 0

#test for 5 rounds of 10 documents
while ptest_rounds < 5:
    testdocs_2 = rand_docs(10,100)
    tokensdocs_2 = token_texts(testdocs_2)
    dict_tokensdocs_2 = dict_bags(tokensdocs_2)
    processed_docs_2 = cosine_pre_processing(dict_tokensdocs_2)
    
    parallel_output_df = mapping_processing(processed_docs_2,2)
    regular_output_df = sim_documents(dict_tokensdocs_2,'cosinenp')
    
    parallel_output = parallel_output_df['cosine similarity'].to_list()
    regular_output = regular_output_df['cosinenp similarity'].to_list()
    
    length = len(parallel_output)
    for i in range (0,length):
        if parallel_output[i] == regular_output[i]:
            no_match_count.append(1)
        else:
            no_match_count.append(0)
    
    ptest_rounds += 1

#calculate percentage consistency
match_percent = sum(no_match_count) / len(no_match_count) * 100
print(f'The percentage of matching values is: {int(match_percent)} %')


# This confirms the parallel computing output is 100% consistent.

# ### All-Pairs Similarity with Parallel Computing: Optimum Processes

# Differing numbers of processes are tested in the parallel processing function to find the optimum number to use for cosine similarity calculations of 10 documents of 100 words.

# In[24]:


#Create 10 docs of 100 words to test
testdocs_10 = rand_docs(10,100)
tokensdocs_10 = token_texts(testdocs_10)
dict_tokensdocs_10 = dict_bags(tokensdocs_10)
processed_docs_10 = cosine_pre_processing(dict_tokensdocs_10)


# In[25]:


#run function on different numbers of cores
x_processes = [x for x in range(1,6)]
y_avtimes_mapping = []

for core in x_processes:
    mean,std,error = time_it(mapping_processing,processed_docs_10,core)
    y_avtimes_mapping.append(mean)


# In[27]:


#plot results
plt.plot(x_processes,y_avtimes_mapping,c='xkcd:soft green',linewidth=3.0)
plt.xlabel("Number of Processes",fontsize=12)
plt.ylabel("Run Time (Seconds)",fontsize=12)
plt.title("Average Run Times of Parallel Computing of Cosine Similarity Using Differing Numbers of Parallel Processes",size=10)
plt.show()


# The results show that 2 processes is optimum for this computer and computations executed.

# # Conclusion

# The analyses show cosine similarity (<i>n</i><sup>1.008</sup>) applied to dense vectors produces better run times than the Jaccard measure (<i>n</i><sup>1.476</sup>). The cosine method is further optimised using numpy, with run time order <i>n</i><sup>0.819</sup>. This relies upon documents being represented as vectors, but even with dictionaries as inputs (<i>n</i><sup>1.044</sup>), the cosine out-performed the Jaccard.
# <br/><br/>
# A parallel computing method showed a further enhancement to run time, even at relatively small numbers and sizes of texts, and therefore should be included in future applications. With this, an optimum number of parallel processes can and should be found.
# <br/><br/>
# There is much opportunity for development of these methods. Firstly, the pool of 5000 words used to generate texts means high homogeneity between documents tested. Testing was performed on documents up to 20,000 in length, meaning they would likely have much higher similarity than bodies of texts in wider applications and empirical run times may be faster in this analysis since the dictionaries reach a maximum length of 5000. Additionally, the inputs of functions were either dictionaries or vectors and therefore had already been processed from native forms. Real world applications would require pre-processing, so future analysis should explore optimising these steps. Overall, the analyses proposes some favourable methods, cosine similarity and parallel computing, as a direction for similarity processing for collections of corpuses.

# ## References 

# 1. Cesare, Silvio., Xiang, Yang. Software Similarity and Classification, 67-68. United Kingdom: Springer, 2012.
# 2. Chun, Wesley J. Core Python Programming. United States: Pearson Education, 2006.
# 3. Muflikhah, Lailil, and Baharum Baharudin. "Document clustering using concept space and cosine similarity measurement." In 2009 International conference on computer technology and development, vol. 1, pp. 58-62. IEEE, 2009.
# 4. Van Der Walt, Stefan, S. Chris Colbert, and Gael Varoquaux. "The NumPy array: a structure for efficient numerical computation." Computing in science & engineering 13, no. 2 (2011): 22-30.
# 5. Grama, Ananth., Karypis, George., Gupta, Anshul., Kumar, Vipin. Introduction to Parallel Computing. Germany: Addison-Wesley, 2003.

# In[ ]:




