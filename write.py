n = 755
with open('/home/poudelas/Downloads/Training_and_evaluation/data/custom/train.txt', 'w') as cout:
    for i in range(n):
            cout.write('./data/custom/Images/'+str(i)+'.png'+ '\n')
