# Пример использования:
# python md.py -n 100 -b 15 -t 1.0 -r 1000 -e NVE -l ps -i velverlet -c berendsen

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time


np.random.seed(379245)

atom = {'Ar':{'mass':39.948,'eps':1.65e-21,'sigma':3.4e-10}} 

a_name = 'Ar'
m = atom[a_name]['mass'] * 1.6747e-27                # масса, [кг]
eps = atom[a_name]['eps']                           # epsilon, [Джоуль]
sigma = atom[a_name]['sigma']                       # sigma в Lennard-Jones Potential, [м]
kB = 1.380e-23                                      # Коэффициент Больцмана, [Джоуль/K]
Na = 6.022e23                                       # Число Авогадро

dt = 0.005                                          # шаг по времени
rc = 2.5                                            # LJ дистанция отсечки
rc2 = rc ** 2
rci = 1/rc
rc6i = rci ** 6
rs = 2.0                                            # potential switch distance
rs2 = rs ** 2      
ec = 4.0*rc6i*(rc6i - 1)                            # LJ энергия отсечки
tau = 1                                             # Параметр связи Брендсена




def initUnifPos(N,L):
    """ созданиие коробки заполненной N частицами с начальными позициями """
    nc = int(np.floor(np.cbrt(N)))                  # получение ячеек
    lc = float(L / nc)                                # длина каждой ячейки
    pos_arr = np.zeros(shape=(N,3))                 # 3*[x,y,z]
    f = open('init_pos.xyz','w')
    f.write(str(N) + "\n" + "LJ particles\n")       # записать число частиц
    
    for i in range(int(N//np.power(nc,3)) + 1):  
        for xx in range(nc):                        
            for yy in range(nc):                    
                for zz in range(nc):              
                    
                    idx = i * np.power(nc, 3) + xx * np.power(nc, 2) + yy * nc + zz
                    
                    if (i==1):                      # следующий слой
                        if (idx >= N):              # остановить заполнение частицы при превышении заданного количества частиц
                            break
                            
                    x = '{:.12e}'.format(np.random.uniform(low=xx * lc + 0.1 * lc, high=(xx + 1) * lc - 0.1 * lc)) 
                    y = '{:.12e}'.format(np.random.uniform(low=yy * lc + 0.1 * lc, high=(yy + 1) * lc - 0.1 * lc)) 
                    z = '{:.12e}'.format(np.random.uniform(low=zz * lc + 0.1 * lc, high=(zz + 1) * lc - 0.1 * lc))                               
                                
                    pos_arr[idx] = [x, y, z]
                    
                    p = "Ar " + str(x) + " " + str(y) + " " + str(z) + "\n"              
                    f.write(str(p))                                        
    f.close()
                                         
    return pos_arr      




def wrapPBC(pos_arr,N,L):
    
    for i in range(N):   
        pos_arr[i] = pos_arr[i] - L * np.floor(pos_arr[i] / L)
    
    return pos_arr




def initVel(N,T):
    """ создание начальных скоростей """
    vel = np.random.randn(N, 3)                     # скорость [x,y,z] всех частиц
    vs = np.zeros(shape=(N, 1))                      # скорость каждой частицы  
    sumv = 0.0
    sumv2 = 0.0

    sumv = np.sum(vel, axis=0) / N                  # скорость центра масс
    sumv2 = np.sum(vel**2) / N

    """ коэффициент масштабирования для скорости, удовлетворяющий МБ"""
    fs = np.sqrt(3 * T / (sumv2))                       # коэффициент масштабирования для скорости, удовлетворяющий MB

    for i in range(N):
        vel[i] = (vel[i] - sumv) * fs                 # нулевая скорость центра масс

    for i in range (N):
        vs[i] = np.sqrt(np.dot(vel[i], vel[i]))   
        
    np.savetxt('initSpeedDist.dat', vs, delimiter=',')  
        
    return vel, vs 




def getForce(ljTail, pos_arr, N, L):
    
    en = 0.0   
    f_arr = np.zeros(shape=(N, 3))
    r_vec = np.zeros(shape=(1, 3))
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            
            r_vec = pos_arr[i] - pos_arr[j]         # r_vec=[rx,ry,rz]
            r_vec = r_vec - L * np.rint(r_vec / L)      # PBC condition            
            r2 = np.sum(r_vec ** 2)               

            if (r2 <= rc2):
                r = np.sqrt(r2)
                ri = 1 / r
                r2i = 1 / r2
                r6i = r2i ** 3
                
                f_vec = 48 * r2i * r6i * (r6i - 0.5) * r_vec
                ee = 4 * r6i * (r6i - 1) - ec    
                                                                              
                f_arr[i] = f_arr[i] + f_vec
                f_arr[j] = f_arr[j] - f_vec                     
                en = en + ee 
        
    return f_arr, en




def integration(ensemble, N, L, T, ljTail, thermostat, integrator, sig_a, r_c, v_c, en, r_old=None, v_old=None, f_c=None):

    
    sumv2 = 0.0    
    ek = 0.0
    etot = 0.0
    en_new = 0.0
    
                                                 # нужно r_old
    # r_new = 2*r_c - r_old + f_c*dt**2          # r(t+dt)
    # v_c = (r_new - r_old)/(2*dt)               # текущая скорость v(t), не использовать v(t) в цикле, сохранить для Ek

    # sumv2 = np.sum(v_c ** 2)
    # temp = sumv2 / (3 * N)                         # 1/2 mv^2 = 3/2 NkT --->T = mv^2/(3Nk)
    # ek = 0.5 * sumv2
    # etot = en + ek
    # r_old = np.copy(r_c)
    # r_c = np.copy(r_new)
    
      
                                               # нужно f_c  
    r_new = r_c + v_c*dt + 0.5*f_c*dt**2       # r(t+dt)       
    v_new = v_c + 0.5*f_c*dt                   # используем f(t), [v(t+dt)=v(t)+(f(t+dt)+f(t))/2m *dt]
    f_new, en_new = getForce(ljTail,r_new,N,L) # новая сила f(t+dt)
    v_new = v_new + 0.5*f_new*dt               # используем f(t+dt)
    
    sumv2 = np.sum(v_c ** 2)
    temp = sumv2 / (3 * N)  
    ek = 0.5 * sumv2
    etot = en + ek  
    r_c = np.copy(r_new)
    v_c = np.copy(v_new)
    f_c = np.copy(f_new) 


    v_c = v_c*np.sqrt(1+dt/tau*(T/temp-1))       
            
        
    return r_c, v_c, r_old, v_old, f_c, en_new, temp, en, ek, etot 



    
def writeCoords(r_c, N, L):

    r_wrap = wrapPBC(r_c, N, L)
    one_traj = []
    one_traj.append(str(N) + "\n" + "LJ частиц\n")    
    for i in range(N):                               
        x = '{:.7f}'.format(r_wrap[i][0]) 
        y = '{:.7f}'.format(r_wrap[i][1]) 
        z = '{:.7f}'.format(r_wrap[i][2])   
        p = "Ar " + str(x) + " " + str(y) + " " + str(z) + "\n"            
        one_traj.append(str(p))  
        
    return one_traj



     
def writeThermoVal(thermo_val, N):
        
    thermo_val[:,0] = thermo_val[:,0]*dt*sigma*np.sqrt(m/eps)*1.0e12   # конвертировать в  ps  
    thermo_val[:,1] = thermo_val[:,1]*eps/kB                           # конвертировать в K
    thermo_val[:,2:] = thermo_val[:,2:]*eps*Na/N*1.0e-3                # конвертировать в [кДжоуль/моль]
    np.savetxt('thermoVal.txt', thermo_val, fmt='%8.3f',header='Время(пс), темп(K), Еп  Ек  Еполн (кДж/моль)')  

    return thermo_val
   

    
    
def plotSpeedDist(vs, T):
       
    bin_wid = 20.0
    vs = vs*np.sqrt(eps/m)      # конвертировать в м/c
    bins = np.arange(min(vs), max(vs) + bin_wid, bin_wid)
    n, bins, patches = plt.hist(vs, bins=bins,  facecolor='r', alpha=0.2, label='T='+str(np.round(T * eps / kB)) + 'K')    
    plt.xlabel('Начальная скорость (м/с)')
    plt.ylabel('Вероятность')
    plt.title('Распределение по скоростям')
    plt.grid(True)
    plt.legend()
#    plt.show()
    plt.savefig('is_distr.png')  
    plt.close()
    

    
    
def plotThermo(thermo_val, ensemble, ljTail, integrator):
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(thermo_val[:,0], thermo_val[:,1],label='T')   # график температуры
    
    plt.legend()
    plt.ylabel('Температура (K)')

    plt.subplot(1, 2, 2)
    plt.plot(thermo_val[:,0], thermo_val[:,2], label='Пот. Энергия')
    plt.plot(thermo_val[:,0], thermo_val[:,3], label='Кин. Энергия')
    plt.plot(thermo_val[:,0], thermo_val[:,4], label='Полн. Энерг')
    plt.ylabel('Энергия (кДж/моль)')
    plt.set_xlabel('Время (пс)')
    plt.legend()
    plt.show()
#    plt.savefig('thermoVal.png')    
#    plt.close()
    
 

    
def main():   
    
    N = 100
    L = 15
    T = 1.0
    nsteps = 1000
    ensemble = 'NVE'
    ljTail = 'ps'
    integrator = 'velverlet' #verlet
    thermostat = 'berendsen'
    sig_a = np.sqrt(T)                     
    print ("\n Симуляция частиц Аргона с потенциалом ЛД\n")
    print ("Число частиц: %s\n" % N)
    print ("Длина бокса: %.1f (sigma)\n" % L)
    print ("Плотность Аргона: %.3f кг/м^3\n" % (m * N / np.power(L * sigma, 3)))

    r_c = initUnifPos(N, L)                        # начальная позиция
    v_c, vs = initVel(N, T)                        # начальные скорости   
    f_c, en = getForce(ljTail, r_c, N, L)            # начальная сила
    v_old = v_c - 0.5 * f_c * dt   
    r_old = r_c - v_c*dt                          # r(t-dt), для Верле метода
    mdTraj = []    
    one_traj = writeCoords(r_c, N, L)  
    mdTraj.append(one_traj)                       # записать траекторию
    
    step = 0  
    thermo_val = []                              
    
    while step < nsteps:  
        
        if (step%100 == 0): 
            print ("step %s" % step)     
                 
        """ Интегрирование """
        r_c, v_c, r_old, v_old, f_c, en_new, temp, en, ek, etot = integration(ensemble, N, L, T, ljTail, thermostat, integrator, sig_a, r_c, v_c, en, r_old=r_old, v_old=v_old, f_c=f_c)

        if (step%10 == 0):   
            
            thermo_val.append([step, temp, en, ek, etot])  
            one_traj = writeCoords(r_c, N, L)  
            mdTraj.append(one_traj)                    
                
                        
        en = en_new #f_c, en = getForce(ljTail,r_c,N,L)        
            
        step = step + 1
            
    thermo_val = np.asarray(thermo_val)
    thermo_val = writeThermoVal(thermo_val, N)
    
    file = open('Traj.xyz', 'w')  
    for i in range(len(mdTraj)):
        for j in range(len(mdTraj[i])):
            file.write(mdTraj[i][j])
    file.close()
    plotThermo(thermo_val, ensemble, ljTail, integrator)  
    plotSpeedDist(vs, T)

  

        
if __name__ == "__main__":
    start_time = time.time()
    main()            
    print("--- Прошло: %.2f секунд ---" % (time.time() - start_time))