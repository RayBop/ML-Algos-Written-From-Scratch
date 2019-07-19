from __future__ import print_function
import sys
import math
#import 3


class Node:
    
  def __init__(self, attribute, value, data, depth, leftchild, rightchild, used_attributes):
    self.attribute = attribute
    self.value = value
    self.data = data
    self.depth = depth
    self.leftChild = leftchild
    self.rightChild = rightchild
    self.classification = None
    self.used_attributes = used_attributes
    
def log2(x):
    if x == 0 :
        return 0
    else:
        return math.log2(x)

def compute_error_rate(lines):
    
    first_row_vals = lines[1].split(",")
    first_label = first_row_vals[len(first_row_vals)-1]
    
    
    count = 0
    total_count = 0
    for i in range(1,len(lines)):
        row_vals = lines[i].split(",")
        label_val = row_vals[len(row_vals)-1]
        if (label_val == first_label):
            count += 1
        total_count += 1
        
        
    x = count / total_count
    
    return min(x, 1-x)
    
def retrieve_attribute_vals(data, feature):
    
    head_row = data[0].split(",")
    
    if feature == "label":
        feature_index = len(head_row)-1
    else:
        feature_index = head_row.index(feature)
        
    vals = []
    i = 0
    for row in data:
        if i == 0:
            i += 1
            continue 
        if len(vals) == 2:
            break
        value = (row.split(","))[feature_index]
        if value not in vals:
            vals.append(value)
    return vals 
        
#need to copy the header row
#need to skip header row in calculations
def split_data(data, decision_attribute, values):
    
    data1, data2 = [], []
    feature_index = (data[0].split(",")).index(decision_attribute)
    
    i = 0
    for row in data:
        row_array = row.split(",")
        if (i==0):
            data1.append(row)
            data2.append(row)
            i+=1
            continue
        if (row_array[feature_index] == values[0]):
            data1.append(row)
        elif (row_array[feature_index] == values[1]):
            data2.append(row)
    #print(len(data), len(data1), len(data2))
    return data1, data2
            
        
def retrieve_best_decision(data, label):
    index = len(data[0].split(","))-1
    vals = retrieve_attribute_vals(data, label)
    #print(vals)
    
    i = 0
    c1,c2 = 0,0
    
    for row in data:
        row = row.split(",")
        if i == 0:
            i += 1
            continue
        if (row[index] == vals[0]):
            c1 += 1
        elif (row[index] == vals[1]):
            c2 += 1
        
    return vals[0] if c1 > c2 else vals[1]
    

def compute_MI (node,feature):
    data = node.data
    head_row = data[0].split(",")
    feature_index = (head_row).index(feature)
    label_index = len(head_row)-1
    F = retrieve_attribute_vals(data, feature)
    Y = retrieve_attribute_vals(data, "label")# 2nd arg == 'label' then retrieve label vals
    #print(F,Y)
    p1, p2, p3, p4, count, = 0,0,0,0,0
    
    
    #compute 4 probabilities (Y=0,B=0  Y=1,B=0  ...) and then use those to 
    #quickly compute entropy and mutual info
    
    i =0
    
    for row in data:
        row = row.split(",")
        
        
        if i == 0:
            i += 1
            continue
        f_val = row[feature_index]
        y_val = row[label_index]
        
        if (f_val == F[0] and y_val == Y[0]):
            p1 += 1
        elif (len(F) == 2 and f_val == F[1] and y_val == Y[0]):
            p2 += 1
        elif (f_val == F[0] and y_val == Y[1]):
            p3 += 1
        elif (len(F) == 2 and f_val == F[1] and y_val == Y[1]):
            p4 += 1
        count += 1
        
        
        
    f0 = (p1+p3)/count
    f1 = 1-f0
    y0 = (p1+p2)/count
    y1 = 1-y0
    
    ## potential division by zero
    
    if f0 == 0:
        y0_given_f0 = 0
        y1_given_f0 = 0
    else :
        y0_given_f0 = (p1/count)/f0
        y1_given_f0 = (p3/count)/f0
        
    if f1 == 0:
        y0_given_f1 = 0
        y1_given_f1 = 0
    else:
        y0_given_f1 = (p2/count)/f1
        y1_given_f1 = (p4/count)/f1
    
    
    
    
    
    
    x1=  -(y0*log2(y0) + y1*log2(y1)) 
    x2 =  (f0*( -(y0_given_f0*log2(y0_given_f0) + y1_given_f0*log2(y1_given_f0)))) 
    x3 =  (f1*( -(y0_given_f1*log2(y0_given_f1) + y1_given_f1*log2(y1_given_f1))))
    
    
    MI = x1 - (x2+x3)
    
    
    return MI
            
        
        
        
        
        
    
  
def find_best_attribute(node, attributes):
    
    best_feature = attributes[0]
    Best_mutual_info = 0
    
    #print(type(attributes))
    for feature in attributes[:-1]:
        #print(feature)
        if (feature not in node.used_attributes):
            #print(feature)
            MI = compute_MI(node,feature)
            #print(MI)
            if (MI> Best_mutual_info):
                Best_mutual_info = MI
                best_feature = feature
    if Best_mutual_info == 0:
        return None
    return best_feature
    
def train_tree(root, max_depth, attributes):
    
    #print(root.depth)
    decision_attribute = find_best_attribute(root, attributes)  #returns none is no mutual info is >0.
    perfectly_classified = compute_error_rate(root.data) == 0
    attributes_used = len(attributes)<= len(root.used_attributes)
    label = attributes[len(attributes)-1]
    #print(decision_attribute == None,perfectly_classified, root.depth >= (int)(max_depth),attributes_used  )
    
    
    if decision_attribute == None or perfectly_classified or root.depth >= (int)(max_depth) or attributes_used:
        #print(decision_attribute == None,perfectly_classified, root.depth >= (int)(max_depth),attributes_used  )
        #pprint.pprint(root.data)
        classification = retrieve_best_decision(root.data, label)
        root.classification = classification 
        #print(classification )
    else:
        root.attribute = decision_attribute
        values = retrieve_attribute_vals(root.data, decision_attribute) #returns 2 element array of vals
        data1, data2 = split_data(root.data, decision_attribute, values) 
        #pprint.pprint(data1)
        #pprint.pprint(data2)
        
        
        root.rightChild = Node(None, values[0], data1, root.depth +1, None, None, root.used_attributes + [decision_attribute] )
        root.leftChild = Node(None, values[1], data2, root.depth +1, None, None, root.used_attributes + [decision_attribute])
        train_tree(root.rightChild, max_depth, attributes)
        train_tree(root.leftChild, max_depth, attributes)
    
    
    ###
    ###
    
    
    
    #decision_attribute = find_best_attribute(root, attributes)# should also consult list of used attrbutes and attributes
    #values = retrieve_attribute_vals(root.data, decision_attribute) #returns 2 element array of vals
    
    #data1, data2 = split_data(root.data, decision_attribute, values) #data1 has data for decision_attribute == values[0]
    
    #root.rightChild = Node(decision_attribute, values[0], data1, root.depth +1, None, None, root.used_attribtes + [decision_attribute] )
    #root.leftChild = Node(decision_attribute, values[1], data2, root.depth +1, None, None, root.used_attribtes + [decision_attribute])
   # 
    #perfectly_classified_right = compute_error_rate(data1) == 0
    #perfectly_classified_left = compute_error_rate(data2) == 0
    #attributes_used_R = len(attributes) > len(root.rightChild.used_attributes)
    #attributes_used_L = len(attributes) > len(root.leftChild.used_attributes)
    
    
    #label = attributes(len(attributes)-1)
    
    #if (not perfectly_classified_right and root.depth < max_depth and not attributes_used_R):
        #train_tree(root.rightChild, max_depth, attributes)
    #else:
        #classification = retrieve_best_decision(root.rightChild.data, label)
        #root.rightChild.classification = classification
    
    #if (not perfectly_classified_left and root.depth < max_depth and not attributes_used_L):
        #train_tree(root.leftChild, max_depth, attributes)
    #else:
        #classification = retrieve_best_decision(root.leftChild.data, label)
        #root.leftChild.classification = classification
    
def predict(row, tree):
    
    row = row.split(",")
    node = tree
    
    while (node.classification == None):
        attribute = node.attribute
        att_index = (node.data[0]).split(",").index(attribute)
        val = row[att_index]
        
        if (val == node.rightChild.value):
            node = node.rightChild
        else:
            node = node.leftChild
            
    return node.classification
        
    
    
    
    
def get_predictions(tree, data):
    
    predictions = []
    i = 0
    for row in data:
        if i == 0:
            i += 1
            continue
        predictions.append(predict(row, tree))
        
    return predictions
        
        
def retrieve_attributes(data):
    
    return (data[0]).split(",")
    
    
def count_vals(data, vals):
    
    i = 0
    v0,v1=0,0
    for row in data:
        if i==0:
            i+=1
            continue
            
        row_array = row.split(",")
        if row_array[len(row_array)-1] == vals[0]:
            v0 += 1
        else:
            v1 += 1
    return v0,v1
        
def print_tree(tree, attribute, label_values):
    #print(tree.attribute)
   
    if (tree != None):
        v0,v1 = count_vals(tree.data, label_values)
        string = ''
        string += tree.depth*'| '
        if attribute != None:
            string += attribute + " = " + tree.value + ": "
        string += "[" + (str)(v0) + ' ' + label_values[0] + '/' + (str)(v1) + ' ' + label_values[1] + ']'
        print(string)
        
        print_tree(tree.rightChild, tree.attribute, label_values)
        print_tree(tree.leftChild, tree.attribute, label_values)
        
        
    
def train(data, max_depth):
    root = Node(None, None, data, 0, None, None, [])
    attributes = retrieve_attributes(root.data)
    train_tree(root, max_depth, attributes)
    return root


if __name__ == ('__main__'):
                        
    #continue training if:
    # 1. highest mutual info is > 0
    # 2. current depth < max_depth
    # 3. there still exist some attrbutes to split on
    
    
    train_input = sys.argv[1] #csv
    test_input = sys.argv[2] #csv 
    max_depth = sys.argv[3]
    train_out = sys.argv[4] #.labels
    test_out = sys.argv[5] #.labels
    metrics_out = sys.argv[6] #.txt
    
    
    
    train_data = open(train_input)
    test_data = open(test_input)
    train_out_label = open(train_out,'w')
    test_out_label = open(test_out,'w')
    metrics = open(metrics_out,'w')
    
    
    
    train_set = (train_data.read()).splitlines()
    test_set = (test_data.read()).splitlines()
    tree = train(train_set, max_depth)
    
    train_predictions = get_predictions(tree, train_set)
    test_predictions = get_predictions(tree, test_set)
    
    
    
    
  
    for prediction in train_predictions:
        train_out_label.write(prediction + '\n')
        
    
    for prediction in test_predictions:
        test_out_label.write(prediction + '\n')
    
    count, errors = 0,0
    label_index = len(train_set[0].split(","))-1
    for i in range(len(train_predictions)):
        if (train_predictions[i] != (train_set[i+1].split(","))[label_index]):
            errors += 1
        count += 1
    
    metrics.write("error(train): " + (str)(errors/count) + '\n')
    
    
    count, errors = 0,0
    label_index = len(test_set[0].split(","))-1
    for i in range(len(test_predictions)):
        if (test_predictions[i] != (test_set[i+1].split(","))[label_index]):
            errors += 1
        count += 1
        
    metrics.write("error(test): " + (str)(errors/count) + '\n')
        
    label_values = retrieve_attribute_vals(train_set, "label")
    print_tree(tree, None, label_values)
    
    
    
    
    #error_rate = compute_error_rate(lines)
    #p = error_rate
    
    #entropy = -p*math.log(p)/math.log(2) - (1-p)*math.log(1-p)/math.log(2)
    
    #for i in range(0,len(lines)):
        #print(lines[i])
        
    #f2.write("entropy: " + str(entropy) + '\n')
    #f2.write("error: " + str(error_rate) + '\n')
       
    
    
    
    train_data.close()
    test_data.close()
