#!/usr/bin/python
import climate
import theanets
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import collections
import random 
import sys


PATH_PREFIX="matR_char"

def myfiles(mypath, myfilter="num"):
    files =[ f for f in listdir(mypath) if isfile(join(mypath,f)) and f.startswith(myfilter) ]
    return files

def max_length(files):
    maxlen = 0;
    for f in files:
        mat = sio.loadmat(join(PATH_PREFIX,f))
        length = len(mat['gest'][1])
        if maxlen < length:
            maxlen = length
    return maxlen

def loadMat(files):
    inputs = collections.defaultdict(dict)
    for f in files:
        splitted = f.split("_")
        category_details=splitted[2:]
        mat = sio.loadmat(join(PATH_PREFIX,f))
        inputs[splitted[1]]["_".join(category_details)] =  dict(x=mat['gest'][1], y=mat['gest'][2], z=mat['gest'][3])
        #print mat['gest'][1]
    return inputs

def findMax(inputs):
    maxlen = 0;
    for key in inputs.keys():
        for category_details in inputs[key]:
            length = len(inputs[key][category_details]['x'])
            if maxlen < length:
                maxlen = length
    return maxlen

def normalize(elem,minimum,maximum):
    return (elem - minimum) / (maximum - minimum)

def extendTo(inputs, maximum):
    for key in inputs.keys():
        for category_details in inputs[key]:
            length = len(inputs[key][category_details]['x'])
            #print inputs[key][category_details]['x']
            mini= min(inputs[key][category_details]['x'])
            maxi= max(inputs[key][category_details]['x'])
            #print "min: {}  max: {} ".format(mini, maxi)
            #print [normalize(elem, mini, maxi) for elem in  inputs[key][category_details]['x']]
            inputs[key][category_details]['x'] = [normalize(elem, mini, maxi) for elem in  inputs[key][category_details]['x']]
            mini= min(inputs[key][category_details]['y'])
            maxi= max(inputs[key][category_details]['y'])
            inputs[key][category_details]['y'] = [normalize(elem, mini, maxi) for elem in  inputs[key][category_details]['y']]
            mini= min(inputs[key][category_details]['z'])
            maxi= max(inputs[key][category_details]['z'])
            inputs[key][category_details]['z'] = [normalize(elem, mini, maxi) for elem in  inputs[key][category_details]['z']]
            for i in range(length, maximum):
                inputs[key][category_details]['x'] = np.append(inputs[key][category_details]['x'], [0])
                inputs[key][category_details]['y'] = np.append(inputs[key][category_details]['y'], [0])
                inputs[key][category_details]['z'] = np.append(inputs[key][category_details]['z'], [0])
            inputs[key][category_details]['all'] = np.append(inputs[key][category_details]['x'], inputs[key][category_details]['y'])
            inputs[key][category_details]['all'] = np.append(inputs[key][category_details]['all'], inputs[key][category_details]['z'])
    return inputs


def to_np(A):
    B = (np.array(A[0]), np.array(A[1], dtype=np.int32))   
    B[0].astype('f')
    B[1].astype('i')
    return B

def create_dataset(extended):
    Tr   = ([],[])
    V    = ([],[])
    Tst  = ([],[])

    for key in sorted(extended.keys()):
        i = 0
        end = len(extended[key])
        for category_details in extended[key]:
            #print (key,ord(key), category_details)
            if (i < end*0.7): 
                Tr[0].append( extended[key][category_details]['all']) 
                Tr[1].append( ord(key))
            elif (i < end*0.9): 
                V[0].append( extended[key][category_details]['all']) 
                V[1].append( ord(key))
            else :
                Tst[0].append( extended[key][category_details]['all']) 
                Tst[1].append( ord(key))
            i+= 1
    return (to_np(Tr), to_np(V), to_np(Tst))

def usage():
    print  "{} <num|lower|upper|all>\n".format(sys.argv[0])

def get_load_type(argv):
    if (argv == "all") : 
        return ""
    else :
       return argv
def handle_args():
    if len(sys.argv) < 2 :
        usage() 
        exit(1)
    if (sys.argv[1] not in ["all", "num", "lower", "upper"]):
        usage() 
        exit(1)
 

def tn_main():
    handle_args()
    load_type = get_load_type(sys.argv[1])
    files = myfiles(PATH_PREFIX, load_type)
    # load the matlab files to a dict
    inputs = loadMat(files)
    # determine the max on all data
    maximum = findMax(loadMat(myfiles(PATH_PREFIX, "")))
    # extend the x y z coordinates
    inputs = extendTo(inputs, maximum)
    # split them into train/validation/test datasets (70,20,10)
    Tr,V,Tst = create_dataset(inputs)

    climate.enable_default_logging()
   # Create a classification dataset.
#X, y = make_classification( n_samples=3000, n_features=100, n_classes=10, n_informative=10)
#exit(0)
#X = X.astype('f')
#y = y.astype('i')
#cut = int(len(X) * 0.8)  # training / validation split
#train = X[:cut], y[:cut]
#valid = X[cut:], y[cut:]
# Build a classifier model with 100 inputs and 10 outputs.
    #exp = theanets.Experiment(theanets.Classifier, layers=(654, (654,'relu') , (654,'relu'), (62,'softmax')))
    layer_input = 3 * maximum
    layer_hidden = layer_input / 2
    layer_output = layer_input / 10
    exp = theanets.Experiment(theanets.Classifier, layers=(layer_input, (layer_hidden, 'relu'), (layer_hidden, 'relu') , (layer_output, 'softmax')))

# Train the model 
    exp.train(Tr, V, learning_rate=1e-4, momentum=0.9, min_improvement=0.01)

# Show confusion matrices on the training/validation splits.
    for label, (X, y) in (('training:', Tr), ('validation:', V), ('test:', Tst)):
        print(label)
        predicted = exp.network.predict(X)
        result = [(chr(f[0]),chr(f[1]))  for f in zip( predicted, y)]
        print sum([ 1 if (a==b) else 0 for a,b in result ]) / float(len(result))
        print "(expected,predicted):"
        print [ (b,a) for a,b in result if a != b]
        print(confusion_matrix(y, predicted))

if __name__ == '__main__':
    tn_main()
