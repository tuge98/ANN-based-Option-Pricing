

from itertools import repeat

lis = ["1","2","3","4"]
lis2 = ["5","6","7","8"]

diff = 10
appendlist =[]



num = 10
list2 = []
for i in range(len(lis)):

    templist = lis2[:i]
    val = 1
    
    #templist.insert(i,num)
    templist.extend(val for i in range(10))
    #list2.append([templist])
    list2.append(templist)
    

print(list2)


    
    




   



#print(appendlist)





