import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter, freqz



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y  # Filter requirements.

def readCSVasFloat(filename):
  lines = open(filename).readlines()
  returnArray = [list(map(float, line.strip().split(' '))) for line in lines]
  returnArray = np.array(returnArray)
  return returnArray

def acceleration(action_sequence):
    action_sequence_cacceleration=action_sequence[2:]+action_sequence[:-2]-2*action_sequence[1:-1]
    return action_sequence_cacceleration

def filter_6hz(action_sequence,order=6,fs=120,cutoff=6):
    filter_out=np.zeros((action_sequence.shape[0],action_sequence.shape[1]))
    for i in range(action_sequence.shape[1]):
       filter_out[:,i]=butter_lowpass_filter(action_sequence[:,i], cutoff, fs, order)
    return filter_out

def distance(action_sequence,point1,point2):
    dis_x=action_sequence[0][3*(point1-1)]-action_sequence[0][3*(point2-1)]
    dis_y=action_sequence[0][3*(point1-1)+1]-action_sequence[0][3*(point2-1)+1]
    dis_z=action_sequence[0][3*(point1-1)+2]-action_sequence[0][3*(point2-1)+2]
    dis=np.sqrt(pow(dis_x,2)+pow(dis_y,2)+pow(dis_z,2))
    return dis

def all_mass(action_sequence):
    mass={}
    llg=[11,10,9,8,7,1]
    rlg=[6,5,4,3,2,1]
    body=[16,15,14,13,1]
    ls=[22,21,19,18,14]
    rs=[30,28,27,26,14]
    distance_lg=0
    distance_shoulder=0
    distance_body=0
    #leg
    for i in range(len(llg)-1):
        distance_lg+=distance(action_sequence,llg[i],llg[i+1])
        dis1=distance(action_sequence,llg[i],llg[i+1])
        mass[int(llg[i])] =dis1
        distance_lg+=distance(action_sequence,rlg[i],rlg[i+1])
        dis2=distance(action_sequence,rlg[i],rlg[i+1])
        mass[int(rlg[i])] =dis2
    #shoulder
    for i in range(len(ls)-1):
        distance_shoulder+=distance(action_sequence,ls[i],ls[i+1])
        dis3=distance(action_sequence,ls[i],ls[i+1])
        mass[int(ls[i])] =dis3
        distance_shoulder+=distance(action_sequence,rs[i],rs[i+1])
        dis4=distance(action_sequence,rs[i],rs[i+1])
        mass[int(rs[i])] =dis4
    #body
    for i in range(len(body)-1):
        distance_body+=distance(action_sequence,body[i],body[i+1])
        dis5=distance(action_sequence,body[i],body[i+1])
        mass[int(body[i])] =dis5
    #Finger
    distance_shoulder+=distance(action_sequence,31,28)+distance(action_sequence,23,21)
    dis6=distance(action_sequence,31,28)
    dis7=distance(action_sequence,23,21)
    mass[int(31)] = dis6
    mass[int(23)] = dis7
    return distance_shoulder+distance_body+distance_lg,mass

def mass_proportion(mass,all_mass,human=70):
    for i in mass:
        mass[i] = mass[i]*human/all_mass
    return mass

def next_force(Fx, Fy, Fz, mass, acc, i):
    Fx_next = Fx-mass[i]*acc[:,3*(i-1)]
    Fy_next = Fy-mass[i]*acc[:,3*(i-1)+1]
    Fz_next = Fz-mass[i]*acc[:,3*(i-1)+2]
    return Fx_next, Fy_next, Fz_next

def save_force(Fx,Fy,Fz,i,Force):
        Force[:,3*(i-1)] = Fx
        Force[:,3*(i-1)+1] = Fy
        Force[:,3*(i-1)+2] = Fz
        return Force
    
def Leg_force(llg,Force,mass,acc):
    Fx,Fy,Fz=0,0,0
    for i in llg:
        Fx = Fx-mass[i]*acc[:,3*(i-1)]
        Fy = Fy-mass[i]*acc[:,3*(i-1)+1]
        Fz = Fz-mass[i]*acc[:,3*(i-1)+2]   
        Force[:,3*(i-1)] = Fx
        Force[:,3*(i-1)+1] = Fy
        Force[:,3*(i-1)+2] = Fz
    return Force

def Shoulder(ls,Force,mass,acc):
    Fx,Fy,Fz=0,0,0
    for i in ls:
        Fx=Fx-mass[i]*acc[:,3*(i-1)]
        Fy=Fy-mass[i]*acc[:,3*(i-1)+1]
        Fz=Fz-mass[i]*acc[:,3*(i-1)+2]   
        Force[:,3*(i-1)]=Fx
        Force[:,3*(i-1)+1]=Fy
        Force[:,3*(i-1)+2]=Fz
        if(i==ls[0] or i==ls[1]):
            Fx=0-mass[i]*acc[:,3*(i-1)]
            Fy=0-mass[i]*acc[:,3*(i-1)+1]
            Fz=0-mass[i]*acc[:,3*(i-1)+2]   
            Force[:,3*(i-1)]=Fx
            Force[:,3*(i-1)+1]=Fy
            Force[:,3*(i-1)+2]=Fz
        if(i==ls[2]):
            Fx=mass[i]*acc[:,3*(ls[0]-1)]+mass[i]*acc[:,3*(ls[1]-1)]-mass[i]*acc[:,3*(i-1)]
            Fy=mass[i]*acc[:,3*(ls[0]-1)]+mass[i]*acc[:,3*(ls[1]-1)+1]-mass[i]*acc[:,3*(i-1)+1]
            Fz=mass[i]*acc[:,3*(ls[0]-1)]+mass[i]*acc[:,3*(ls[1]-1)+2]-mass[i]*acc[:,3*(i-1)+2]   
            Force[:,3*(i-1)]=Fx
            Force[:,3*(i-1)+1]=Fy
            Force[:,3*(i-1)+2]=Fz
    return Fx,Fy,Fz,Force

def Shoulder_link(F1,F2,F3,F4,F5,F6,F7,F8,F9,mass,acc,i):
     Fx=F1+F4+F7-mass[i]*acc[:,3*(i-1)]
     Fy=F2+F5+F8-mass[i]*acc[:,3*(i-1)+1]
     Fz=F3+F6+F9-mass[i]*acc[:,3*(i-1)+2]
     return Fx,Fy,Fy
def Bodyf(Fx5,Fy5,Fz5,Fx6,Fy6,Fz6,body,Force,mass,acc):
    Fx,Fy,Fz=0,0,0
    for i in body:
        Fx=Fx-mass[i]*acc[:,3*(i-1)]
        Fy=Fy-mass[i]*acc[:,3*(i-1)+1]
        Fz=Fz-mass[i]*acc[:,3*(i-1)+2]   
        Force[:,3*(i-1)]=Fx
        Force[:,3*(i-1)+1]=Fy
        Force[:,3*(i-1)+2]=Fz
        if (i==body[2]):
            Fx,Fy,Fz=Shoulder_link(Fx,Fy,Fz,Fx5,Fy5,Fz5,Fx6,Fy6,Fz6,mass,acc,i)
            Force[:,3*(i-1)]=Fx
            Force[:,3*(i-1)+1]=Fy
            Force[:,3*(i-1)+2]=Fz
    return Force

def same_force(x,y,Force):
    Force[:,3*(x-1)]=Force[:,3*(y-1)]
    Force[:,3*(x-1)+1]=Force[:,3*(y-1)+1]
    Force[:,3*(x-1)+2]=Force[:,3*(y-1)+2]
    return Force

def Force_cal(mass,acc,add_dim=True):
    llg=[11,10,9,8,7]
    rlg=[6,5,4,3,2]
    body=[16,15,14,13]
    ls=[22,23,21,19,18]
    rs=[30,31,28,27,26]
    Force=np.zeros((acc.shape[0],acc.shape[1]))
    Force=Leg_force(llg,Force,mass,acc)
    Force=Leg_force(rlg,Force,mass,acc)
    Fx5,Fy5,Fz5,Force=Shoulder(ls,Force,mass,acc)
    Fx6,Fy6,Fz6,Force=Shoulder(rs,Force,mass,acc)
    Force=Bodyf(Fx5,Fy5,Fz5,Fx6,Fy6,Fz6,body,Force,mass,acc)
    if add_dim==True:
        for i in [[17,14],[25,14],[24,23],[29,28],[20,21],[29,28],[32,31]]:
            same_force(i[0],i[1],Force)
    Force1=filter_6hz(Force)

    return Force1
 
        
def Force(action_sequence,human=1):

    # action_sequence = readCSVasFloat(action_sequence)
    #acc= acceleration(action_sequence)   
    acc= acceleration(action_sequence) 
   # y=filter_6hz(acc)
    all_mass1,mass=all_mass(action_sequence)
    mass=mass_proportion(mass,all_mass1,human)
    Force_out=Force_cal(mass,acc)
    out=np.zeros((action_sequence.shape[0],action_sequence.shape[1]))
    out[:Force_out.shape[0],:]=Force_out
    out[Force_out.shape[0]:,:]=Force_out[-1]
    return out



