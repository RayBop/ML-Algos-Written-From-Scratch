from __future__ import print_function
import sys
import math




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


if __name__ == ('__main__'):
                        
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    
    
    f1 = open(infile)
    f2 = open(outfile,'w')
    
    lines = (f1.read()).splitlines()
    
    error_rate = compute_error_rate(lines)
    p = error_rate
    
    entropy = -p*math.log(p)/math.log(2) - (1-p)*math.log(1-p)/math.log(2)
    
    #for i in range(0,len(lines)):
        #print(lines[i])
        
    f2.write("entropy: " + str(entropy) + '\n')
    f2.write("error: " + str(error_rate) + '\n')
        
    
    
    
    f1.close()
    f2.close()



        
        
    