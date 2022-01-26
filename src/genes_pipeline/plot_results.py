
from cProfile import label
import sys
import matplotlib.pyplot as plt


x = [2,3,4,5,6,7,8,9,10]
if True:
    #bin size = 4, static

    #PErformance with stopping value = 1
    a = [0.8014981273408239, 0.9525593008739076, 0.9650436953807741, 0.9700374531835206, 0.968789013732834, 0.9625468164794008, 0.9612983770287141, 0.9662921348314607, 0.9588014981273408]

    #PErformance with stopping value = 10
    c = [0.8114856429463171, 0.9575530586766542, 0.9662921348314607, 0.9575530586766542, 0.9563046192259675, 0.9500624219725343, 0.9588014981273408, 0.9563046192259675, 0.9438202247191011]

    #PErformance with stopping value = 20
    d = [0.8127340823970037, 0.951310861423221, 0.9538077403245943, 0.9500624219725343, 0.9662921348314607, 0.9500624219725343, 0.9488139825218477, 0.9413233458177278, 0.9388264669163545]
   
    plt.plot(x, [x*100 for x in a], label = "min size = 1", marker='o')
    plt.plot(x, [x*100 for x in c], label = "min size = 10", marker='^')
    plt.plot(x, [x*100 for x in d], label = "min size = 20", marker='s')

    plt.legend()

    plt.xlabel("N Principal components")
    plt.ylabel("Test accuracy in %")

    plt.title("Test accuracy for different PC \n bin size = 4 (static)")

    plt.ylim([69,100])

    plt.legend()
    plt.grid()
    plt.show()

if True:
    #bin size = 8, static

    #PErformance with stopping value = 1
    a = [0.9026217228464419, 0.951310861423221, 0.9525593008739076, 0.9413233458177278, 0.9400749063670412, 0.9288389513108615, 0.9313358302122348, 0.9225967540574282, 0.9338327091136079]

    #PErformance with stopping value = 10
    c = [0.9026217228464419, 0.9463171036204744, 0.9525593008739076, 0.9338327091136079, 0.9463171036204744, 0.9350811485642946, 0.9300873907615481, 0.9101123595505618, 0.9001248439450686]
    
    #PErformance with stopping value = 20
    d = [0.6966292134831461, 0.8776529338327091, 0.9588014981273408, 0.9588014981273408, 0.9625468164794008, 0.9425717852684145, 0.9325842696629213, 0.9250936329588015, 0.9338327091136079]
    
    plt.plot(x, [x*100 for x in a], label = "min size = 1", marker='o')
    plt.plot(x, [x*100 for x in c], label = "min size = 10", marker='^')
    plt.plot(x, [x*100 for x in d], label = "min size = 20", marker='s')

    plt.legend()

    plt.xlabel("N Principal components")
    plt.ylabel("Test accuracy in %")

    plt.title("Test accuracy for different PC \n bin size = 8 (static)")
    plt.ylim([69,100])

    plt.legend()
    plt.grid()
    plt.show()

if True:
    #bin size = [ n_PC +1, n_PC, n_PC -1, ... , 2], adaptive

    #pc [2,...,10] bins = 8, stoppingCriteria ="size", stoppingValue = 1, k = 10 
    a = [0.6953807740324595, 0.8838951310861424, 0.9625468164794008, 0.968789013732834, 0.9662921348314607, 0.9588014981273408, 0.9588014981273408, 0.9563046192259675, 0.9563046192259675]

    #pc [2,...,10] bins = 8, stoppingCriteria ="size", stoppingValue = 20, k = 10 
    c = [0.7028714107365793, 0.8813982521847691, 0.9637952559300874, 0.9662921348314607, 0.9662921348314607, 0.9525593008739076, 0.9450686641697877, 0.9575530586766542, 0.9538077403245943]

    #PErformance with stopping value = 20
    d = [0.6966292134831461, 0.8776529338327091, 0.9588014981273408, 0.9588014981273408, 0.9625468164794008, 0.9425717852684145, 0.9325842696629213, 0.9250936329588015, 0.9338327091136079]
    
    plt.plot(x, [x*100 for x in a], label = "min size = 1", marker='o')
    plt.plot(x, [x*100 for x in c], label = "min size = 10", marker='^')
    plt.plot(x, [x*100 for x in d], label = "min size = 20", marker='s')

    plt.legend()

    plt.xlabel("N Principal components")
    plt.ylabel("Test accuracy in %")

    plt.title("Test accuracy for different PC \n bin size is adaptive")
    plt.ylim([69,100])

    plt.legend()
    plt.grid()
    plt.show()
