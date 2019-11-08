import subprocess 
import numpy
import math
import matplotlib.pyplot as plt
import random
import os
import shutil
import re
import time
import keras
from sklearn.externals import joblib
from ripser import ripser
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve,validation_curve
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor as GBR
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution1D,MaxPooling1D,AveragePooling1D
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_pdf import PdfPages

def readposcar(target_pos):
    pos = []
    f = open(target_pos)
    try:
        for line in f:
            pos.append(line)
    except:
        f.close()
    lattice = []
    pos_all=[]
    for item in pos[2:5]:
        try:
            lattice.append(list(map(float, item.split())))
        except:
            return False
    for item in lattice:
        if len(item) != 3: return False
    #for item in pos[6]:
     #   try:
      #      orig_atom_num=list(map(float, item.split()))
       # except:
        #    return False 
    for item in pos[8:]:
        try:
            pos_all.append(list(map(float, item.split())))
        except:
            return False 
    return([lattice,pos_all])
    

def getsuperpoint(lattice,pos,N):
    ID=0
    superpoint=[]
    superID=[]
    for xi in range(N):
        superID.append([])
        for yi in range(N):
            superID[xi].append([])
            for zi in range(N):
                superID[xi][yi].append([])
                for posi in range(len(pos)):
                    tempx=pos[posi][0]*lattice[0][0]+pos[posi][1]*lattice[1][0]+pos[posi][2]*lattice[2][0]+xi*lattice[0][0]+yi*lattice[1][0]+zi*lattice[2][0]
                    tempy=pos[posi][0]*lattice[0][1]+pos[posi][1]*lattice[1][1]+pos[posi][2]*lattice[2][1]+xi*lattice[0][1]+yi*lattice[1][1]+zi*lattice[2][1]
                    tempz=pos[posi][0]*lattice[0][2]+pos[posi][1]*lattice[1][2]+pos[posi][2]*lattice[2][2]+xi*lattice[0][2]+yi*lattice[1][2]+zi*lattice[2][2]
                    superpoint.append([tempx,tempy,tempz])
                    superID[xi][yi][zi].append(ID)
                    ID=ID+1
    return superpoint,superID
        
def makeposcar(target_pos,lattice,pos):
    f = open(target_pos,'w')
    f.write('by code\n1.0\n')
    for item in lattice:
        f.write('%15.8f %15.8f %15.8f\n' % tuple(item))
    f.write('Li \n%d\ndirect\n'%len(pos))
    for item in pos:
        f.write('%15.8f %15.8f %15.8f\n' % tuple(item))
    f.close()

def point2poscar(target_pos,point):
    LATIICE_n=round(len(point)/2)+15
    lattice_tmp=[[LATIICE_n,0,0],[0,LATIICE_n,0],[0,0,LATIICE_n]]
    tem_point=[]
    for item in point:
        tem_point.append(list(map(lambda x:x/LATIICE_n+0.5,item)))
    makeposcar(target_pos,lattice_tmp,tem_point)
def pos_vibra(pos,model):
#    beste=0
#    return pos,beste
    buchang=(1/far_x)*1.5
    atom_num=len(pos)
    betti_all=[]
    newpos=list(range(len(pos)))
    for ji in range(len(pos)):
        newpos[ji]=list(pos[ji])
    timer=0
    goon=1
    bestpos=list(pos)
    whole_time=0
    oldbetti=pos2betti([pos])
    betti_all.append(oldbetti[0])
#    oldbetti[0].append(round(far_old[0]*far_k)/far_k)
    olde=model.predict(numpy.array(oldbetti))
    beste=olde[0]
    while goon==1:
        whole_time=whole_time+1
        for i_p in range(len(pos)):
            newpos[i_p]=list([pos[i_p][0]+random.random()*random.randint(-1,1)*buchang , pos[i_p][1]+random.random()*random.randint(-1,1)*buchang , pos[i_p][2]+random.random()*random.randint(-1,1)*buchang ])
        newbetti=pos2betti([newpos])
        if newbetti[0] in betti_all:
            print('dup')
            continue
        betti_all.append(newbetti[0])
        oldbetti=pos2betti([pos])
#        far_new=pos_farest_dis([newpos])
#        far_old=pos_farest_dis([pos])
#        newbetti[0].append(round(far_new[0]*far_k)/far_k)
#        oldbetti[0].append(round(far_old[0]*far_k)/far_k)
        if  newbetti[0]== oldbetti[0]:
            
            newpos=list(pos)
            continue
        newe=model.predict(numpy.array(newbetti))
        olde=model.predict(numpy.array(oldbetti))
#        print('old:%f  new:%f'%(olde[0],newe[0]))
        
        if (newe[0]-olde[0])<=0:
            timer=0
            pos=list(newpos)
            if beste>newe[0]:
                beste=newe[0]
                bestpos=list(newpos)
            continue
        else:               
            
            if math.exp(-(newe[0]-olde[0])*2625*atom_num)>random.random() and timer<50:  #1 A.U.=1 hatree = 2625.500 KJ/mol
                print(math.exp(-(newe[0]-olde[0])*2625))
                pos=list(newpos)
                continue     
            else:
#                print(math.exp(-(newe[0]-olde[0])))
                timer=timer+1
                newpos=list(pos)
        if timer>atom_num*10:
            print('done---------------timer>50-----------------------------------------------------------')
            print(whole_time)
            break
#            buchang=buchang*2
#            timer=0
            
#            if buchang>xbuchang:
#                break
            
        if whole_time>atom_num*10000:
            print('done---------------whole_time>10000---------------------------------------------------')
            print(whole_time)
            break
    return bestpos,beste
#def pos_vibra(pos,model):
#    buchang=(1/far_x)*1.5
#    atom_num=len(pos)
#    newpos=list(range(len(pos)))
#    for ji in range(len(pos)):
#        newpos[ji]=list(pos[ji])
#    timer=0
#    goon=1
#    bestpos=list(pos)
#    whole_time=0
#    oldbetti=pos2betti([pos])
#    far_old=pos_farest_dis([pos])
##    oldbetti[0].append(round(far_old[0]*far_k)/far_k)
#    olde=model.predict(oldbetti)
#    beste=olde[0]
#    while goon==1:
#        whole_time=whole_time+1
#        for i_p in range(len(pos)):
#            newpos[i_p]=list([pos[i_p][0]+random.random()*random.randint(-1,1)*buchang , pos[i_p][1]+random.random()*random.randint(-1,1)*buchang , pos[i_p][2]+random.random()*random.randint(-1,1)*buchang ])
#            newbetti=pos2betti([newpos])
#            oldbetti=pos2betti([pos])
#    #        far_new=pos_farest_dis([newpos])
#    #        far_old=pos_farest_dis([pos])
#    #        newbetti[0].append(round(far_new[0]*far_k)/far_k)
#    #        oldbetti[0].append(round(far_old[0]*far_k)/far_k)
#            if  newbetti[0]== oldbetti[0]:
#                newpos=list(pos)
#                continue
#            newe=model.predict(newbetti)
#            olde=model.predict(oldbetti)
#    #        print('old:%f  new:%f'%(olde[0],newe[0]))
#            if (newe[0]-olde[0])<=0:
#                timer=0
#                pos=list(newpos)
#                if beste>newe[0]:
#                    beste=newe[0]
#                    bestpos=list(newpos)
#                continue
#            else:               
#                
#                if math.exp(-(newe[0]-olde[0])*2625500)>random.random() and timer<50:  #1 A.U.=1 hatree = 2625.500 KJ/mol
#    #                print(math.exp(-(newe[0]-olde[0])))
#                    pos=list(newpos)
#                    continue     
#                else:
#    #                print(math.exp(-(newe[0]-olde[0])))
#                    timer=timer+1
#                    newpos=list(pos)
#        if timer>atom_num*20:
#            print('done---------------timer>50-----------------------------------------------------------')
#            print(whole_time)
#            break
##            buchang=buchang*2
##            timer=0
#            
##            if buchang>xbuchang:
##                break
#            
#        if whole_time>atom_num*1000:
#            print('done---------------whole_time>10000---------------------------------------------------')
#            print(whole_time)
#            break
#    return bestpos,beste
def pos_pred_ene(pos,model):
    oldbetti=pos2betti([pos])
#    far_old=pos_farest_dis([pos])
#    oldbetti[0].append(round(far_old[0]*far_k)/far_k)
    olde=model.predict(oldbetti)
    print(olde)
    return olde
    
def dist_matric(pos):    
    dis_mat=[]
    if len(pos)<100:
        for def_i in range(1,len(pos)):
            for def_j in range(def_i):
                temp=math.sqrt(math.pow(pos[def_j][0]-pos[def_i][0],2)+math.pow(pos[def_j][1]-pos[def_i][1],2)+math.pow(pos[def_j][2]-pos[def_i][2],2))
                dis_mat.append(temp)
    dis_mat.sort()
    return(dis_mat)
def dist_matric2(pos):    
    dis_mat=[]
    for def_i in range(0,len(pos)):
        dis_mat.append([])
        for def_j in range(0,len(pos)):
            temp=math.sqrt(math.pow(pos[def_j][0]-pos[def_i][0],2)+math.pow(pos[def_j][1]-pos[def_i][1],2)+math.pow(pos[def_j][2]-pos[def_i][2],2))
            dis_mat[def_i].append(temp)
    return(dis_mat)
def pos_center_dist_mat(pos_tmp,c_mat):
    c_mat1=[]
    for i_2 in range(len(pos_tmp)):
        tmp_dis= math.pow(pos_tmp[i_2][0]-c_mat[0],2)+math.pow(pos_tmp[i_2][1]-c_mat[1],2)+math.pow(pos_tmp[i_2][2]-c_mat[2],2) 
        c_mat.append(tmp_dis)
    c_mat1.sort()
    return c_mat
    
def check_dup(pos1,pos2,limit=0.1):
#    pos1_tmp,c1_pmat=pos2standard_coord([pos1])
#    pos2_tmp,c2_pmat=pos2standard_coord([pos2])
#    c1_mat=pos_center_dist_mat(pos1_tmp[0],[0,0,0])
#    c2_mat=pos_center_dist_mat(pos2_tmp[0],[0,0,0])
#    c1_mat.sort()
#    c2_mat.sort()
    pos1_mat=dist_matric(pos1)   
    pos2_mat=dist_matric(pos2) 
    for def_i2 in range(len(pos1_mat)):
        if abs(abs(pos1_mat[def_i2]) - abs(pos2_mat[def_i2]))>limit:
            return False
    return True
def check_dup2(pos1_list,pos2,limit=0.1):
    for item1 in range(len(pos1_list)):
        if check_dup(pos1_list[item1],pos2,limit):
            return True
    return False
def too_quick(v):
    if abs(v)>0.5*atom_distances:
        return v/abs(v)*0.5*atom_distances
    else:
        return v


def too_close(pos,dis=False):
    min_dis=math.pow(atom_distances-1.5,2)#-0.8
    if dis:
        min_dis=math.pow(dis,2)
    for i in range(1,len(pos)):
        for j in range(i):
            if  math.pow(pos[i][0]-pos[j][0],2)+math.pow(pos[i][1]-pos[j][1],2)+math.pow(pos[i][2]-pos[j][2],2) <min_dis:
                return True
    return False
def too_far(pos,dis=False):
    max_dis=math.pow(atom_distances+1,2)
    if dis:
        max_dis=math.pow(dis,2)
    tmp_list=[pos[0]]
    for k in range(len(pos)):
        for i in tmp_list:
            for j in pos:
                if  math.pow(i[0]-j[0],2)+math.pow(i[1]-j[1],2)+math.pow(i[2]-j[2],2) < max_dis and j not in tmp_list:#+0.3
                    tmp_list.append(j)
    if len(tmp_list)!=len(pos):

        return True
    else:
        return False

def search_close_point(ID,exit_ID,N_super,label='FCC'):    #!only for FCC, and 0 is z_low atom, 1 is z_high atom!  #only for FCC  #only for FCC  #only for FCC  #only for FCC  #only for FCC 
    #print('search_close_point begin')
    if label == 'FCC':
        close_ID1=[ [ID[0]+1,ID[1],ID[2],ID[3]] , [ID[0],ID[1]+1,ID[2],ID[3]] , [ID[0]+1,ID[1]+1,ID[2],ID[3]] , [ID[0]-1,ID[1],ID[2],ID[3]] , [ID[0],ID[1]-1,ID[2],ID[3]], [ID[0]-1,ID[1]-1,ID[2],ID[3]] ]
        if ID[3]==0:
            close_ID2=[ [ID[0],ID[1],ID[2],ID[3]+1] , [ID[0],ID[1]+1,ID[2],ID[3]+1] , [ID[0]-1,ID[1],ID[2],ID[3]+1] ]
            close_ID3=[ [ID[0],ID[1],ID[2]-1,ID[3]+1] , [ID[0],ID[1]+1,ID[2]-1,ID[3]+1] , [ID[0]-1,ID[1],ID[2]-1,ID[3]+1] ]
        if ID[3]==1:
            close_ID2=[ [ID[0],ID[1],ID[2],ID[3]-1] , [ID[0],ID[1]-1,ID[2],ID[3]-1] , [ID[0]+1,ID[1],ID[2],ID[3]-1] ]
            close_ID3=[ [ID[0],ID[1],ID[2]+1,ID[3]-1] , [ID[0],ID[1]-1,ID[2]+1,ID[3]-1] , [ID[0]+1,ID[1],ID[2]+1,ID[3]-1] ]
        close_ID=close_ID1 + close_ID2 + close_ID3
        for i in range(len(close_ID)):
            close_ID[i]=[close_ID[i][0]%N_super,close_ID[i][1]%N_super,close_ID[i][2]%N_super,close_ID[i][3]]
    if label == 'BCC':
        if  ID[3]==0:
            close_ID1=[ [ID[0],ID[1],ID[2],1] , [ID[0]-1,ID[1],ID[2],1] , [ID[0],ID[1]-1,ID[2],1] , [ID[0]-1,ID[1]-1,ID[2],1] ]
            close_ID2=[ [ID[0],ID[1],ID[2]-1,1] , [ID[0]-1,ID[1],ID[2]-1,1] , [ID[0],ID[1]-1,ID[2]-1,1] , [ID[0]-1,ID[1]-1,ID[2]-1,1] ]
        if  ID[3]==1:
            close_ID1=[ [ID[0],ID[1],ID[2],0] , [ID[0]+1,ID[1],ID[2],0] , [ID[0],ID[1]+1,ID[2],0] , [ID[0]+1,ID[1]+1,ID[2],0] ]
            close_ID2=[ [ID[0],ID[1],ID[2]+1,0] , [ID[0]+1,ID[1],ID[2]+1,0] , [ID[0],ID[1]+1,ID[2]+1,0] , [ID[0]+1,ID[1]+1,ID[2]+1,0] ]
        close_ID=close_ID1 + close_ID2 
        for i in range(len(close_ID)):
            close_ID[i]=[close_ID[i][0]%N_super,close_ID[i][1]%N_super,close_ID[i][2]%N_super,close_ID[i][3]]
    x_try=0
    while x_try<7:
        x_try=x_try+1
        if label == 'FCC':
            No_ID_temp=random.randint(0,11)      #!  first_close atom number
        if label == 'BCC':
            No_ID_temp=random.randint(0,7)      #!  first_close atom number            
        if close_ID[No_ID_temp] not in exit_ID:
            #print('search_close_point done')
            return close_ID[No_ID_temp]
    return []

def make_vasp(target,point,if_run,if_opt=True,if_pred=False):
    file_Kpt=target+'/KPOINT'
    file_INC=target+'/INCAR'
    file_POS=target+'/POSCAR'
    with open( file_Kpt,'w' ) as fk:
        fk.write('AUTO GRID\n0\nG\n1 1 1\n0 0 0')
    with open( file_INC,'w' ) as fi:
        fi.write('SYSTEM=PSO\nLWAVE=.FALSE.\nLCHARG=.FALSE.\nISTART=0\nALGO=Fast\nENCUT=500\nLORBIT=11\nISMEAR=0\nPREC=Normal\n')
        fi.write('NELMIN=3\nPOTIM=0.2\nISIF=2\nEDIFF=0.1E-3\nEDIFFG=-1.0E-2\nLREAL = .False.\nNCORE= 4\nSIGMA=0.1\n')
        fi.write('NELM=30\n')
        if if_opt and not if_pred:
            fi.write('NSW=20\nIBRION=2\nISPIN=1\n')
        if if_opt and if_pred:
            fi.write('NSW=100\nIBRION=3\nISPIN=2\n')
        else:
            fi.write('NSW=0\n')
    point2poscar(file_POS,point)
    subprocess.call('cp pbs-vasp %s'%target,shell=True)
    subprocess.call('cp POTCAR %s'%target,shell=True)
    if if_run:
        orig_os=os.getcwd()
        os.chdir(target)
        subprocess.call('qsub pbs-vasp',shell=True)
        os.chdir(orig_os)
        
        
def make_gaus(target,point,if_run,pred=False,nomaxcycle=True,opt=False):
    len_point=len(point)
    if pred==True:
        mag_range=[1]    
    elif len_point<12:
        if len_point%2==0 and pred==False:
            mag_range=list(range(1,len_point+1,2))
        elif len_point%2==1 and pred==False:
            mag_range=list(range(2,len_point+1,2))
    elif len_point>=12:
        if len_point%2==0 and pred==False:
            mag_range=list(range(1,round(len_point/3)+2,2))
        elif len_point%2==1 and pred==False:
            mag_range=list(range(2,round(len_point/3)+2,2))
#    if len_point%2==0 :
#        mag_range=[1]
#    else:
#        mag_range=[2]

    for i in mag_range:
        file_calc_2=target+'/'+'%d'%i
        os.mkdir(file_calc_2)
        file_gau=file_calc_2+'/dmac.gjf'
        with open( file_gau,'w' ) as fg:
            if not opt:
                fg.write('%chk=C.chk\n%mem=10GB\n\n# bpw91/6-31G pop=full nosym scf=(maxcycle=80,xqc)\n\npso\n\n')
            if opt:
                fg.write('%chk=C.chk\n%mem=10GB\n\n# bpw91/6-31G pop=full opt=(maxstep=50,maxcycle=50,loose) nosym scf=(maxcycle=80,xqc)\n\npso\n\n')
            fg.write('0 %d\n'%i)
            for item in point:
                fg.write('Li %15.8f %15.8f %15.8f\n' % tuple(item))
            fg.write('\n')
        subprocess.call('cp gaupbs %s'%file_calc_2,shell=True)
        if if_run:
            orig_os=os.getcwd()
            os.chdir(file_calc_2)
            subprocess.call('qsub gaupbs',shell=True)
            os.chdir(orig_os)
def if_cal_done_vasp(target,pop_size):
    try:
        if_done=True
        for i in range(1,pop_size+1):
            vasp_out=target+'/'+'%d'%i+'/OUTCAR'
            vasp_out2=target+'/'+'%d'%i+'/vasp.out'
            vpo=open(vasp_out,'r')
            vpread=vpo.read()
            vpo2=open(vasp_out2,'r')
            vpread2=vpo2.read()
            if_1=re.search('CPU time used',vpread)
            if_2=re.search('reached required',vpread)
            if_3=re.search('please rerun with',vpread2)
            if if_1==None and if_2==None  and if_3==None:
                print('not done %s'%vasp_out)
                if_done=False
                return False
            print('done %s'%vasp_out)
                
        if if_done==True:
            return True
    except:
        return False
def if_cal_done_gaus(target,pop_size):
    if_done=True
    for i in range(1,pop_size+1):

        try:
            #print(1)
            target1=target+'/'+'%d'%i
            mag_list=os.listdir(target1)
         #   print(2)
            for j in mag_list:
                target2=target1+'/'+j
                gaus_log=target2+'/'+'DMAC.log'
                with open(gaus_log,'r') as gl:
                    glread=gl.read()
                    print(i)
                    print(j)
                    if_1=re.search('Job cpu time',glread)
                    if_2=re.search('termination',glread)
                    if if_1==None and if_2==None:
                        if_done=False
                        return False
        except:
#            print('i')
            return False
    if if_done==True:
        return True       




def random_point_gen(atom_num,pos_all,posID_all,add_how_much,label):
    #print('random_point_gen begin')
    if label == 'BCC':
        lattice,pos=readposcar('POSCAR0')
    if label == 'FCC':
        lattice,pos=readposcar('POSCAR1')
    if atom_num<10:
        N_super=round(atom_num*3)
    else:
        N_super=round(atom_num)
    supercell,superID=getsuperpoint(lattice,pos,N_super)
    center_atom=round((N_super-1)/2)
    origi_posall_len=len(pos_all)
    while len(pos_all)-origi_posall_len<add_how_much:
        pos=[]
        pos_real=[]
        while len(pos)<atom_num:
            if len(pos)==0:
                pos.append([center_atom,center_atom,center_atom,0])
                pos_real.append(supercell[superID[center_atom][center_atom][center_atom][0]])
            else:
                len_pos=len(pos)
                mom_atom=random.randint(0,(len_pos-1))
                temp=search_close_point(pos[mom_atom],pos,N_super,label)
                if temp==[]:
                    continue
                temp_real=supercell[superID[temp[0]][temp[1]][temp[2]][temp[3]]]
                #print(len(pos_all))
                #print(len_pos)
                pos.append(temp)
                pos_real.append(temp_real)
        add_pos=True
        for i in range(len(pos_all)):

            if check_dup(pos_all[i],pos_real):
                add_pos=False
        if add_pos:
            
            posID_all.append(pos)
#            pos_real=make_pos_fit(pos_real)
            pos_all.append(pos_real)
    pos_all,cc_tmp=pos2standard_coord(pos_all)
    
    #print('random_point_gen done')
    return pos_all,posID_all
def analysis_vasp(target,atom_num):
    pos=[]
    energy=[]
    mag=[]
    pop_list=os.listdir(target)

    for i1 in pop_list:
        orig_os=os.getcwd()
        pos_4=[]
        energy_4=[]
        mag_4=[]
        target1=target+'/'+i1
        if_there_is_normal=False
        gaus_log=target1+'/'+'vasp.out'
        xdat_log=target1+'/'+'XDATCAR'
        del_list=[]
        os.chdir(target1)
        with open('vasp.out','r') as df:
            dfread=df.read()
            match_tmp=re.findall(r"RMM.*\n.*F=.*", dfread)
            for rs in range(len(match_tmp)):
                match_tmp1=match_tmp[rs].split()
                if int(match_tmp1[1])>29:
                    del_list.append(rs)
                energy_4.append(float(match_tmp1[11]))
                mag_4.append(float(0))
        with open('XDATCAR','r') as xf:
            xfread=xf.read()
            match_tmp2=xfread.split('\n')
            x1_tmp=match_tmp2[2].split()
            y1_tmp=match_tmp2[3].split()
            z1_tmp=match_tmp2[4].split()
            x1=float(x1_tmp[0])
            y1=float(y1_tmp[1])
            z1=float(z1_tmp[2])
            match_dir=re.findall(r"Direct configuration", xfread)
            for pop_i in range(len(match_dir)):
                pos_tmp2=[]
                for item1 in range(atom_num):
                    read_line=8+pop_i*(atom_num+1)+item1
                    pos_tmp3=match_tmp2[read_line].split()
                    for j3 in range(3):
                        pos_tmp3[j3]=float(pos_tmp3[j3])-0.5
                    
                    pos_tmp2.append([ round(float(pos_tmp3[-3]),4)*x1,round(float(pos_tmp3[-2]),4)*y1,round(float(pos_tmp3[-1]),4)*z1 ])
                pos_4.append(pos_tmp2)
        while len(pos_4)<len(energy_4):
            del energy_4[-1]
            del mag_4[-1]            
        for del_k in range(len(del_list)-1,-1,-1):
            try:
                del pos_4[del_list[del_k]]
                del energy_4[del_list[del_k]]
                del mag_4[del_list[del_k]]
            except:
                pass

        os.chdir(orig_os)
        pos=pos+pos_4
        energy=energy+energy_4
        mag=mag+mag_4
    
    #print(energy.index(min(energy)))
    return pos,energy,mag   
def analysis_gaus(target,atom_num):
    pos=[]
    energy=[]
    mag=[]
    pop_list=os.listdir(target)
    for i1 in pop_list:
        pos_4=[]
        energy_4=[]
        mag_4=[]
        target1=target+'/'+i1
        mag_list=os.listdir(target1)
        if_there_is_normal=False
        for j1 in mag_list:
            target2=target1+'/'+j1
            gaus_log=target2+'/'+'DMAC.log'
            with open(gaus_log,'r') as gl:
                glread=gl.read()
                if_1=re.search('Normal termination of Gaussian',glread)
                if if_1!=None:
                    if_there_is_normal=True
                    break
        #print(i1)
#        print(if_there_is_normal)
        planB_ene=[]
        for j1 in mag_list:
            target2=target1+'/'+j1
            gaus_log=target2+'/'+'DMAC.log'
            with open(gaus_log,'r') as gl:
                glread=gl.read()
                if_1=re.search('Normal termination of Gaussian',glread)
                if if_1!=None:
                    orig_os=os.getcwd()
                    os.chdir(target2)            
                    pos_tmp2=[]
                    with open('DMAC.log','r') as df:
                        dfread=df.read()
                        match_tmp=re.findall(r"SCF Done.*", dfread)
                        match_tmp2=match_tmp[-1].split()
                        if int(match_tmp2[-2]) !=81:
                            e_tmp2=float(match_tmp2[4])
                            match_tmp3=re.findall(r"Input ori.*Distance", dfread,re.S)
                            match_tmp4=match_tmp3[-1].split('\n')
                            match_tmp5=match_tmp4[-atom_num-2:-2]
                            for item1 in range(atom_num):
                                pos_tmp3=match_tmp5[item1].split()
                                pos_tmp2.append([ round(float(pos_tmp3[-3]),2),round(float(pos_tmp3[-2]),2),round(float(pos_tmp3[-1]),2) ])
                            pos_4.append(pos_tmp2)
                            energy_4.append(e_tmp2)
                            mag_4.append(j1)
                    os.chdir(orig_os)
                elif if_1==None and if_there_is_normal==False:
                    try:
                        orig_os=os.getcwd()
                        os.chdir(target2)            
                        pos_tmp2=[]
                        with open('DMAC.log','r') as df:
                            dfread=df.read()
                            match_tmp=re.findall(r"SCF Done.*", dfread)
                            match_tmp2=match_tmp[-1].split()
                            if int(match_tmp2[-2]) !=81:
                                e_tmp2=float(match_tmp2[4])
                                match_tmp3=re.findall(r"Input ori.*Distance", dfread,re.S)
                                match_tmp4=match_tmp3[-1].split('\n')
                                match_tmp5=match_tmp4[-atom_num-2:-2]
                                for item1 in range(atom_num):
                                    pos_tmp3=match_tmp5[item1].split()
                                    pos_tmp2.append([ round(float(pos_tmp3[-3]),2),round(float(pos_tmp3[-2]),2),round(float(pos_tmp3[-1]),2) ])
                                pos_4.append(pos_tmp2)
                                energy_4.append(e_tmp2)
                                mag_4.append(j1)
                        os.chdir(orig_os)
                    except:
                        os.chdir(orig_os)
        if len(energy_4)==0:
            energy_4=planB_ene
        try:
            pos_5,energy_5,mag_5=pos_ener_sort(pos_4,energy_4,mag_4)
            pos.append(pos_5[0])
            energy.append(energy_5[0])
            mag.append(mag_5[0])
        except:
            pass
    
    #print(energy.index(min(energy)))
    return pos,energy,mag
def analysisdeep_gaus(target,atom_num):
    pos1f=[]
    energy1f=[]
    mag1f=[]

    cycle_rec=[]

    qdel_list=[]
    pop_list=os.listdir(target)
    for i1 in pop_list:
        target1=target+'/'+'%s'%i1
        mag_list=os.listdir(target1)
        
#        if atom_num%2==0:
#            mag_list=['1']
#        else:
#            mag_list=['2']
#        
        
        for j1 in mag_list:
            target2=target1+'/'+j1
            gaus_log=target2+'/'+'DMAC.log'
            with open(gaus_log,'r') as gl:
                glread=gl.read()
                if_1=re.search('Normal termination of Gaussian',glread)
                pos=[]
                energy=[]
                mag=[]
                if if_1!=None:
                    orig_os=os.getcwd()
                    os.chdir(target2)                               
                    e_tmp=subprocess.check_output("grep 'SCF Do' DMAC.log|awk '{print $5,$8}'",shell=True,universal_newlines=True)
                    e_tmp=e_tmp.split('\n')
#                    print(len(e_tmp)-1)
                    for item3 in range(len(e_tmp)-1):
                        e_tmp2_1=e_tmp[item3].split()
                        if int(e_tmp2_1[1]) != 81:
                            energy.append(float(e_tmp2_1[0]))
                            cycle_rec.append(int(e_tmp2_1[1]))
                            mag.append(int(j1))
                    pos_tmp=subprocess.check_output(" sed '/Input orientation:/,/Distance matrix/!d' DMAC.log| grep -v -E 'I|C|N|D|R|\-\-\-'",shell=True,universal_newlines=True)
                    pos_tmp=pos_tmp.split('\n')
                    cycle_num=round((len(pos_tmp)-1)/(atom_num))
#                    print(cycle_num-1)
#                    if cycle_num-1!=len(e_tmp)-1:
#                        print('attention!!!: %s  %s'%(target1,target2))
                    for item2 in range(0,cycle_num-1):
                        pos_tmp2=[]
                        for item1 in range(atom_num):
                            pos_tmp3=pos_tmp[item1+item2*(atom_num)].split()
                            pos_tmp2.append([ round(float(pos_tmp3[-3]),2),round(float(pos_tmp3[-2]),2),round(float(pos_tmp3[-1]),2) ])
                        pos.append(pos_tmp2)                        
                    os.chdir(orig_os)
                else:
                    try:
                        orig_os=os.getcwd()
                        os.chdir(target2)                               
                        e_tmp=subprocess.check_output("grep 'SCF Do' DMAC.log|awk '{print $5,$8}'",shell=True,universal_newlines=True)
                        e_tmp=e_tmp.split('\n')
#                        print(len(e_tmp)-1)
                        for item3 in range(len(e_tmp)-1):
                            e_tmp2_1=e_tmp[item3].split()
                            if int(e_tmp2_1[1]) != 81:
                                energy.append(float(e_tmp2_1[0]))
                                cycle_rec.append(int(e_tmp2_1[1]))
                                mag.append(int(j1))

                        pos_tmp=subprocess.check_output(" sed '/Input orientation:/,/Distance matrix/!d' DMAC.log| grep -v -E 'I|C|N|D|R|\-\-\-'",shell=True,universal_newlines=True)
                        pos_tmp=pos_tmp.split('\n')
                        cycle_num=round((len(pos_tmp)-1)/(atom_num))
#                        print(cycle_num-1)
#                        if cycle_num-1!=len(e_tmp)-1:
#                            print('attention!!!: %s  %s'%(target1,target2))
                        for item2 in range(0,cycle_num-1):
                            pos_tmp2=[]
                            for item1 in range(atom_num):
                                pos_tmp3=pos_tmp[item1+item2*(atom_num)].split()
                                pos_tmp2.append([ round(float(pos_tmp3[-3]),2),round(float(pos_tmp3[-2]),2),round(float(pos_tmp3[-1]),2) ])
                            pos.append(pos_tmp2)                        
                        os.chdir(orig_os)  
                    except:
                        pass
                if len(pos) == len(energy):
                    pos1f=pos1f+pos
                    energy1f=energy1f+energy
                    mag1f=mag1f+mag
#                elif len(pos) == len(energy)-1:
#                    pos1f=pos1f+pos[0:-1]
#                    energy1f=energy1f+energy
#                    mag1f=mag1f+mag
#                print(target2)
#                print(len(pos))
#                print(len(energy))
    return pos1f,energy1f,mag1f

def pos_ener_sort(pos,energy,mag):
    if len(pos)==len(energy):
        item_num=len(energy)
        for i1 in range(item_num):
            for i2 in range(1,item_num):
                if energy[i2-1]>energy[i2]:
                    energy_tmp=energy[i2-1]
                    energy[i2-1]=energy[i2]
                    energy[i2]=energy_tmp
                    pos_tmp=pos[i2-1]
                    pos[i2-1]=pos[i2]
                    pos[i2]=pos_tmp
                    mag_tmp=mag[i2-1]
                    mag[i2-1]=mag[i2]
                    mag[i2]=mag_tmp                
    else:
        print('no equal')
    return pos,energy,mag

def calc_bonding_energy(energy,atom_num,if_even):
    re_energy=[]
    for i in range(len(energy)):
        ene_tmp=(energy[i]-atom_num*sole_energy)
        if if_even==1:
            ene_tmp=ene_tmp/atom_num
            re_energy.append(ene_tmp)
        else:
            re_energy.append(ene_tmp)
    return re_energy

def ini_pos(atom_num,pop_size,cal_method,if_run):
    os.chdir((work_file+'/'+'%d'%atom_num))
    posID_all=[]
    pos_all=[]
    add_how_much=pop_size
    add_how1=round(add_how_much/2)
    add_how2=add_how_much-add_how1
    #print(add_how1)
    #print(add_how2)
    pos_all1,posID_all1=random_point_gen(atom_num,[],[],add_how1,'BCC')
    pos_all2,posID_all2=random_point_gen(atom_num,[],[],add_how2,'FCC')
    #print(pos_all1)
    #print(pos_all2)
    pos_all=pos_all1+pos_all2
    posID_all=posID_all1+posID_all2
   # for i in range(1,len(pos_all)+1):
    #    file_poscar='POSCAR_%d'%i
     #   point2poscar(file_poscar,pos_all[i-1])
    file_calc=('0')
    if os.path.exists( file_calc):
        subprocess.call('rm -r 0',shell=True)
    os.mkdir(file_calc)
    if cal_method=='vasp':
        for i in range(1,len(pos_all)+1):
            file_calc_1=(file_calc+'/%d'%i)
            os.mkdir(file_calc_1)
            make_vasp(file_calc_1,pos_all[i-1],if_run)

    if cal_method=='gaus':
        for i in range(1,len(pos_all)+1):
            if random.randint(0,1):
                opt1=True
            else:
                opt1=False
            file_calc_1=(file_calc+'/%d'%i)
            os.mkdir(file_calc_1)
            make_gaus(file_calc_1,pos_all[i-1],if_run,opt=opt1)
            check_if_sleep()
    os.chdir(work_file)
    return pos_all
def plot_barcode(pos,name): #pos1=[[3,3,0],[3,0,0],[0,3,0],[0,0,0],[1.5,1.5,2.121315],[1.5,1.5,-2.121315]]
    dis_m=dist_matric2(pos)
    ripser_tmp=ripser(numpy.array(dis_m),maxdim=2,thresh=10,distance_matrix=True)
    ripser_tmp2=ripser_tmp.get('dgms')
    betti0_tmp=ripser_tmp2[0]
    betti1_tmp=ripser_tmp2[1]
    betti2_tmp=ripser_tmp2[2]
    fig=plt.figure(dpi=300, figsize=(5, 3))
    ax=[1,2,3,4]
    ax[0] = fig.add_subplot(411)
    ax[1] = fig.add_subplot(412)
    ax[2] = fig.add_subplot(413)
    ax[3] = fig.add_subplot(414)
#    pdf = PdfPages((work_file+'/fig.pdf'))
    
    line_n=1     
    for i2 in range(len(betti0_tmp)):
        if betti0_tmp[i2,1]>10:
            betti0_tmp[i2,1]=10
        betti0_tmp[i2,1]=betti0_tmp[i2,1]-betti0_tmp[i2,0]
        ax[0].broken_barh([(betti0_tmp[i2,0],betti0_tmp[i2,1])],(line_n,0.4),facecolors='tab:purple')
        line_n=line_n+1
    ax[0].set_xlim(0,6)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylim([0,line_n+0.2])
    ax[0].invert_yaxis()  # labels read top-to-bottom[

    line_n=1       
    for i2 in range(len(betti1_tmp)):
        if betti1_tmp[i2,1]>10:
            betti1_tmp[i2,1]=10
        betti1_tmp[i2,1]=betti1_tmp[i2,1]-betti1_tmp[i2,0]
        ax[1].broken_barh([(betti1_tmp[i2,0],betti1_tmp[i2,1])],(line_n,0.4),facecolors='tab:blue')
        line_n=line_n+1
    ax[1].set_xlim(0,6)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    if line_n<5:
        line_n=5
        ax[1].set_ylim([-2,line_n-2+0.2])
    else:
        ax[1].set_ylim([0,line_n+0.2])
    ax[1].invert_yaxis()  # labels read top-to-bottom[
    
    line_n=1    
    for i2 in range(len(betti2_tmp)):
        if betti2_tmp[i2,1]>10:
            betti2_tmp[i2,1]=10
        betti2_tmp[i2,1]=betti2_tmp[i2,1]-betti2_tmp[i2,0]
        ax[2].broken_barh([(betti2_tmp[i2,0],betti2_tmp[i2,1])],(line_n,0.4),facecolors='tab:orange')
        line_n=line_n+1
    ax[2].set_xlim(0,6)
    ax[2].set_xticks([])
    if line_n<5:
        line_n=5
        ax[2].set_ylim([-2,line_n-2+0.2])
    else:
        ax[2].set_ylim([0,line_n+0.2])
    ax[2].set_ylim([0,line_n+0.2])
    ax[2].set_yticks([])
    ax[2].invert_yaxis()  # labels read top-to-bottom[

    line_n=1    
    dis_bar=[]
    for i5 in range(len(dis_m)-1):
        for i6 in range(i5+1,len(dis_m)):
            dis_bar.append([0,dis_m[i5][i6]])
    dis_bar.sort()
    for i3 in range(len(dis_bar)):
        ax[3].broken_barh([(dis_bar[i3][0],dis_bar[i3][1])],(line_n,0.4),facecolors='tab:red')
        line_n=line_n+1
    ax[3].set_xlim(0,6)
    ax[3].set_yticks([])
    if line_n<5:
        line_n=5
        ax[3].set_ylim([-2,line_n-2+0.2])
    else:
        ax[3].set_ylim([0,line_n+0.2]) 
    ax[3].set_ylim([0,line_n+0.2])
    ax[3].invert_yaxis()  # labels read top-to-bottom[
    
    plt.savefig(name)
    plt.show()
#    pdf.close()


def pos2betti(pos):
    betti=[]
    for i1 in range(len(pos)):
        dis_m=dist_matric2(pos[i1])
        ripser_tmp=ripser(numpy.array(dis_m),maxdim=2,distance_matrix=True)
        ripser_tmp2=ripser_tmp.get('dgms')
        betti0_tmp=ripser_tmp2[0]
        betti1_tmp=ripser_tmp2[1]
        betti2_tmp=ripser_tmp2[2]
        betti0=[]
        betti1=[]
        betti2=[]
        betti3=[]
        betti0_ini=[]
        betti1_ini=[]
        betti2_ini=[]
        betti0_fin=[]
        betti1_fin=[]
        betti2_fin=[]
        i_thresh_mat=list(map(lambda x:x/far_x,range(0,far_x*10+1,1)))
        for i_thresh in range(len(i_thresh_mat)-1):
            betti0.append(0)
            betti1.append(0)
            betti2.append(0)
            betti3.append(0)
            betti0_ini.append(0)
            betti1_ini.append(0)
            betti2_ini.append(0)
            betti0_fin.append(0)
            betti1_fin.append(0)
            betti2_fin.append(0)
            for i2 in range(len(betti0_tmp)):
                if i_thresh_mat[i_thresh] >= betti0_tmp[i2][0] and betti0_tmp[i2][1]>=i_thresh_mat[i_thresh+1]:
                    betti0[i_thresh]=betti0[i_thresh]+1
                elif i_thresh_mat[i_thresh] < betti0_tmp[i2][1] and betti0_tmp[i2][1]<i_thresh_mat[i_thresh+1]:
                    betti0[i_thresh]=betti0[i_thresh]+1
            for i3 in range(len(betti1_tmp)):
                if (i_thresh_mat[i_thresh] >= betti1_tmp[i3][0] and betti1_tmp[i3][1]>=i_thresh_mat[i_thresh+1]):
                    betti1[i_thresh]=betti1[i_thresh]+1
                elif (i_thresh_mat[i_thresh] < betti1_tmp[i3][0] and betti1_tmp[i3][1]<i_thresh_mat[i_thresh+1]):
                    betti1[i_thresh]=betti1[i_thresh]+1
                elif (i_thresh_mat[i_thresh] < betti1_tmp[i3][0] and betti1_tmp[i3][0]<i_thresh_mat[i_thresh+1]):
                    betti1[i_thresh]=betti1[i_thresh]+1
                elif (i_thresh_mat[i_thresh] < betti1_tmp[i3][1] and betti1_tmp[i3][1]<i_thresh_mat[i_thresh+1]):
                    betti1[i_thresh]=betti1[i_thresh]+1
            for i4 in range(len(betti2_tmp)):
                if i_thresh_mat[i_thresh] >= betti2_tmp[i4][0] and betti2_tmp[i4][1]>=i_thresh_mat[i_thresh+1]:
                    betti2[i_thresh]=betti2[i_thresh]+1
                elif i_thresh_mat[i_thresh] < betti2_tmp[i4][0] and betti2_tmp[i4][1]<i_thresh_mat[i_thresh+1]:
                    betti2[i_thresh]=betti2[i_thresh]+1
                elif i_thresh_mat[i_thresh] < betti2_tmp[i4][0] and betti2_tmp[i4][0]<i_thresh_mat[i_thresh+1]:
                    betti2[i_thresh]=betti2[i_thresh]+1
                elif i_thresh_mat[i_thresh] < betti2_tmp[i4][1] and betti2_tmp[i4][1]<i_thresh_mat[i_thresh+1]:
                    betti2[i_thresh]=betti2[i_thresh]+1

            for i5 in range(len(dis_m)-1):
                for i6 in range(i5+1,len(dis_m)):
                    if i_thresh_mat[i_thresh] < dis_m[i5][i6] <=i_thresh_mat[i_thresh+1] :
                        betti3[i_thresh]=betti3[i_thresh]+1
#            for i5 in range(len(dis_m)-1):
#                for i6 in range(i5+1,len(dis_m)):
#                    if dis_m[i5][i6] >i_thresh_mat[i_thresh+1] :
#                        betti3[i_thresh]=betti3[i_thresh]+1
#            for i7 in range(len(betti0_tmp)):
#                if i_thresh_mat[i_thresh] <= betti0_tmp[i7][0] and betti0_tmp[i7][0] < i_thresh_mat[i_thresh+1]:
#                    betti0_ini[i_thresh]=betti0_ini[i_thresh]+1
#            for i8 in range(len(betti0_tmp)):
#                if i_thresh_mat[i_thresh] <= betti0_tmp[i8][1] and betti0_tmp[i8][1] < i_thresh_mat[i_thresh+1]:
#                    betti0_fin[i_thresh]=betti0_fin[i_thresh]+1
#            for i9 in range(len(betti1_tmp)):
#                if i_thresh_mat[i_thresh] <= betti1_tmp[i9][0] and betti1_tmp[i9][0] < i_thresh_mat[i_thresh+1]:
#                    betti1_ini[i_thresh]=betti1_ini[i_thresh]+1
#            for i10 in range(len(betti1_tmp)):
#                if i_thresh_mat[i_thresh] <= betti1_tmp[i10][1] and betti1_tmp[i10][1] < i_thresh_mat[i_thresh+1]:
#                    betti1_fin[i_thresh]=betti1_fin[i_thresh]+1
#            for i11 in range(len(betti2_tmp)):
#                if i_thresh_mat[i_thresh] <= betti2_tmp[i11][0] and betti2_tmp[i11][0] < i_thresh_mat[i_thresh+1]:
#                    betti2_ini[i_thresh]=betti2_ini[i_thresh]+1
#            for i12 in range(len(betti2_tmp)):
#                if i_thresh_mat[i_thresh] <= betti2_tmp[i12][1] and betti2_tmp[i12][1] < i_thresh_mat[i_thresh+1]:
#                    betti2_fin[i_thresh]=betti2_fin[i_thresh]+1
#        betti.append(betti0_ini+betti0+betti0_fin+betti1_ini+betti1+betti1_fin+betti2_ini+betti2+betti2_fin+betti3)
##        betti.append(betti0_ini+betti0_fin+betti1_ini+betti1_fin+betti2_ini+betti2_fin+betti3)
        betti.append(betti0+betti1+betti2+betti3)
    return betti
def pos_farest_dis(pos):
    pos_dis=[]
    for i in range(len(pos)):
        dis_mat=dist_matric(pos[i])
        pos_dis.append(dis_mat[-1])
    return pos_dis

def betti_clean(betti_tmp,energy_tmp,mag_tmp):
    del_list=[]
    max_e=max(energy_tmp)
    for i1 in range(1,len(betti_tmp)):
        for i2 in range(0,i1):
            if betti_tmp[i1]==betti_tmp[i2] and energy_tmp[i1]>=energy_tmp[i2]:
                if i1 not in del_list:
                    del_list.append(i1)
            elif betti_tmp[i1]==betti_tmp[i2] and energy_tmp[i1]<energy_tmp[i2]:
                if i2 not in del_list:
                    del_list.append(i2)
#        if 1 < energy_tmp[i1]:
#            if i1 not in del_list:
#                del_list.append(i1)
    del_list.sort() 
    len_tmp=len(del_list)            
    for i3 in range(len_tmp):
        del betti_tmp[del_list[len_tmp-1-i3]]
        del energy_tmp[del_list[len_tmp-1-i3]]
        del mag_tmp[del_list[len_tmp-1-i3]]
    return betti_tmp,energy_tmp,mag_tmp
def self_learn(target_list,pop_size,mingeneration1=49,mingeneration2=1):
    os.chdir(work_file)
    for file_tmp in target_list:
        atom_num=int(file_tmp )
        os.chdir(file_tmp)
#            if atom_num<5:
#                pop_size1=math.pow(atom_num,2)
#            else:
#                pop_size1=pop_size
        ini_pos(atom_num,pop_size,'vasp',True)                                 
        continue_pos(atom_num,'vasp',True,mingeneration1)   
#        genelist=range()
        feature_train,feature_test,target_train,target_test,model,RMSE=get_train_model([file_tmp],['0',str(mingeneration1)],0.0)
        for i_p in range(mingeneration2):               
            continue_pos(atom_num,'vasp',True,1,mingeneration1+i_p,model)
            
            os.chdir(file_tmp)
#                pos_tmp,energy_tmp,mag_tmp=analysis_gaus(str(mingeneration1+i_p+1),atom_num)
#                betti_tmp=pos2betti(pos_tmp)
#                far_tmp=pos_farest_dis(pos_tmp)
#                for j3 in range(len(betti_tmp)):
#        #                betti_tmp[j3].append(round(far_tmp[j3],2))
#                    betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
#                betti_tmp,energy_tmp,mag_tmp=betti_clean(betti_tmp,energy_tmp,mag_tmp)
#                feature_train=feature_train+betti_tmp
#                label='energy'
#                if label=='energy':
#                    energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=0)
#                    target_train=target_train+energy_tmp
#                mag_tmp=list(range(len(target_train)))
#                feature_train,target_train,mag_tmp=betti_clean(feature_train,target_train,mag_tmp)
#                feature_train,target_train=shuffle(feature_train,target_train)
#                model.fit(feature_train,target_train)
    return feature_train,target_train,model
               #feature_train,target_train,model= self_learn(['4'],20,mingeneration1=40,mingeneration2=10)
        
#feature_train,feature_test,target_train,target_test,model,RMSE=get_train_model(['3','4','5'],['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40'],0.3)
def xlim_show(xlim,x1,y1):
    x=list(x1)
    y=list(y1)
    if type(x)==numpy.ndarray:
        x.tolist()
    if type(y)==numpy.ndarray:
        y.tolist()
    for i in range(len(x)-1,-1,-1):
        if x[i]<xlim[0] or x[i]>xlim[1]:
            del x[i]
            del y[i]
    
    plt.scatter(x,y)
    me_1=metrics.r2_score(x,y)
    print(me_1)
    return x,y
def pack_materials(target_list,rank_list1):
    feature_train=[]
    target_train=[]
    label='energy'    #feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','1','2','3','4','5','6','7','8','9'],0.3)
    os.chdir(work_file)
    betti_far_tmp=[]
    all_feature2=[]
    for file_tmp in target_list:          
        atom_num=int(file_tmp )
        feature_train1=[]
        target_train1=[]
        os.chdir(file_tmp)
        rank_list=list(range(int(rank_list1[0]),int(rank_list1[1])+1))
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue

            betti_tmp = list(map(lambda x:x.flatten().tolist(),list(map(numpy.array,pos_tmp))))
 
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train1=feature_train1+betti_tmp
            if label=='energy':
                target_train1=target_train1+energy_tmp
        mag_tmp=list(range(len(target_train1)))
        feature_train1,target_train1,mag_tmp=betti_clean(feature_train1,target_train1,mag_tmp)
        feature_train=feature_train+feature_train1
        target_train=target_train+target_train1
        os.chdir(work_file)

    if True:
        atom_num=20
        os.chdir(str(atom_num))
        rank_list=[2]
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp = list(map(lambda x:x.flatten().tolist(),list(map(numpy.array,pos_tmp))))
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train=feature_train+betti_tmp
            if label=='energy':
                target_train=target_train+energy_tmp
    os.chdir(work_file)
    if True:
        atom_num=40
        os.chdir(str(atom_num))
        rank_list=[2]
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp = list(map(lambda x:x.flatten().tolist(),list(map(numpy.array,pos_tmp))))
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train=feature_train+betti_tmp
            if label=='energy':
                target_train=target_train+energy_tmp
    os.chdir(work_file)
    for i in range(len(feature_train)):
        all_feature2.append([ [len(feature_train[i])/3] + feature_train[i] + [target_train[i]]])

    filesi=open('all_si_test21.py','w') 
    for i in range(len(all_feature2)):
        print(all_feature2[i])
        filesi.write(' '.join('%.4f'%itemp for itemp in all_feature2[i][0]))
        filesi.write('\n')
    filesi.close() 
#    all_feature2=numpy.array(all_feature2)
#    return all_feature2
#    numpy.savetxt('all_si_test',all_feature2,)
        
def get_train_model(target_list,rank_list1,split_ratio=0.3,model_pick='GBR',epo=3,batc=50,plus=False):  #feature_train,feature_test,target_train,target_test,model,RMSE=get_train_model(['2','3','4','5'],['0'],0.3)
    feature_train=[]
    target_train=[]
    label='energy'    #feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','1','2','3','4','5','6','7','8','9'],0.3)
    os.chdir(work_file)
    betti_far_tmp=[]
    for file_tmp in target_list:          
        atom_num=int(file_tmp )
        feature_train1=[]
        target_train1=[]
        os.chdir(file_tmp)
        rank_list=list(range(int(rank_list1[0]),int(rank_list1[1])+1))
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp=pos2betti(pos_tmp)
            print(betti_tmp[0][0])
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train1=feature_train1+betti_tmp
            if label=='energy':
                target_train1=target_train1+energy_tmp
        mag_tmp=list(range(len(target_train1)))
        feature_train1,target_train1,mag_tmp=betti_clean(feature_train1,target_train1,mag_tmp)
        feature_train=feature_train+feature_train1
        target_train=target_train+target_train1
        os.chdir(work_file)
        print(len(target_train))
        print('\\\\\\\\\\\\\\\\\\')
        print(len(feature_train))
    if plus==True:
        atom_num=20
        os.chdir(str(atom_num))
        rank_list=[0,1]
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp=pos2betti(pos_tmp)
            print(betti_tmp[0][0])
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train=feature_train+betti_tmp
            if label=='energy':
                target_train=target_train+energy_tmp
    if model_pick=='GBR':
        gbr=GBR(loss='ls',learning_rate=0.001,n_estimators=15000,max_depth=7,min_samples_split=5,subsample=1.0,max_features='sqrt',random_state=0)
    elif model_pick=='NN':
        gbr=make_dense_model()
    elif model_pick=='NNnf':
        gbr=make_dense_model()
#        gbr = KerasRegressor(build_fn=make_dense_model,nb_epoch=epo,batch_size=batc)
#    print('kk')
    scaler_mm=MinMaxScaler()
    target_train=numpy.array(target_train)
    target_train=target_train.reshape(len(target_train),1)
#    target_train=scaler_mm.fit_transform(target_train)
#    target_train=target_train.tolist()
#    target_train=target_train[0]

    if split_ratio>0.001:
        feature_train1,feature_test1,target_train1,target_test1 =train_test_split(
        feature_train,target_train,test_size=0.3,random_state=0,shuffle=True)    
        if model_pick=='GBR':
            gbr.fit(feature_train1,target_train1)
        elif  model_pick=='NN':
            feature_train1=numpy.array(feature_train1)
            feature_test1=numpy.array(feature_test1)
            target_train1=target_train1.ravel()
            target_test1=target_test1.ravel()
            gbr.fit(feature_train1,target_train1, epochs=epo, batch_size=batc,validation_data=(feature_test1,target_test1),shuffle=True)
        elif model_pick=='NNnf':
            feature_train=numpy.array(feature_train)
            target_train=target_train.ravel()
            kfold = KFold(n_splits=10, shuffle=True, random_state=7)
            
#            results = cross_val_score(gbr, feature_train, target_train, cv=kfold,scoring='neg_mean_absolute_error')
            cvscores = []
            for train, test in kfold.split(feature_train, target_train):
                gbr.fit(feature_train[train],target_train[train], epochs=epo, batch_size=batc,validation_data=(feature_train[test],target_train[test]))
                scores = gbr.evaluate(feature_train[test],target_train[test], verbose=0)
                cvscores.append(scores[1] )
            print('result_mean')
            print(numpy.mean(cvscores))
            return feature_train,None,feature_train,None,None,None,None
        pred_test=gbr.predict(feature_test1)
        print(model_pick)
        print('test MAE:%f'%metrics.mean_absolute_error(target_test1,pred_test))
        RMSE=metrics.mean_squared_error(target_test1,pred_test)
        Pea=pearsonr(target_test1.flatten(),pred_test.flatten())
        pred_train=gbr.predict(feature_train1)
        print('train r2:%f'%metrics.r2_score(target_train1,pred_train))
        pred_test=gbr.predict(feature_test1)
        print('test r2:%f'%metrics.r2_score(target_test1,pred_test))
        print('test pearson:%f'%Pea[0])
        plt.figure(dpi=300, figsize=(3, 3))
        plt.xlim(-1.2,2.2)
        plt.ylim(-1.2,2.2)
        plt.xticks([-1,0,1,2])
        plt.yticks([-1,0,1,2])

     
        x=[-2,-1,0,1,2,3]
        
        plt.plot(target_test1,pred_test,'o',markersize=2)
#        plt.show()
        timenow=time.strftime("%m%d", time.localtime())
        plt.savefig('cv%s%s%d'%(timenow,model_pick,far_x))
        return feature_train1,feature_test1,target_train1,target_test1,gbr,RMSE,scaler_mm
    elif split_ratio<=0.001:
        feature_train1,target_train1 =shuffle(feature_train,target_train)
        feature_test1=[]
        target_test1=[]
        if model_pick=='gbr':
            gbr.fit(feature_train1,target_train1)
        elif  model_pick=='dense':
            feature_train1=numpy.array(feature_train1)
#            feature_test1=numpy.array(feature_test1)
            target_train1=target_train1.ravel()
#            target_test1=target_test1.ravel()
            gbr.fit(feature_train1,target_train1, epochs=epo, batch_size=batc,shuffle=True)
        RMSE='null'
        pred_train=gbr.predict(feature_train1)
        print('train:%f'%metrics.r2_score(target_train1,pred_train))
        return feature_train1,feature_test1,target_train1,target_test1,gbr,RMSE,scaler_mm
def save_feature(feature_train,target_train,feature_test=[],target_test=[],name='feature_target'):
    os.chdir(work_file)   #save_feature(feature_train,target_train,feature_test,target_test,name='feature_target1')

    if type(feature_train)!=list:
        feature_train=feature_train.tolist()
    if type(feature_test)!=list:
        feature_test=feature_test.tolist()
    if type(target_train)!=list:
        target_train=target_train.tolist()
    if type(target_test)!=list:
        target_test=target_test.tolist()
    all_feature=list(feature_train)+list(feature_test)
    all_target=list(target_train)+list(target_test)
    for i_a in range(len(all_feature)):
         all_feature[i_a].append(all_target[i_a])
    all_feature2=numpy.array(all_feature)
    numpy.savetxt(name,all_feature2,fmt="%.3f")

#def evlau_model(model):
    
def load_feature(name='feature_target',split_ratio=0.3,epo=3,batc=50,plus=False):
    os.chdir(work_file)   #s
    xload = numpy.loadtxt(name,delimiter=' ')
    fea_long = len( xload[0] )     #feature_train,feature_test,target_train,target_test,model,RMSE,pea=load_feature(name='feature_target',split_ratio=0.3)
    feature_train = xload[:,0:fea_long-1]
    target_train = xload[:,fea_long-1]
    model_pick='dense'
    if plus==True:
        atom_num=20
        os.chdir(str(atom_num))
        rank_list=[0]
        print(rank_list)
        for ji in rank_list:
            ji=str(ji)
            print(ji)
            if os.path.exists(str(ji)):
                pos_tmp,energy_tmp,mag_tmp=analysis_vasp(ji,atom_num)
            else:
                continue
            betti_tmp=pos2betti(pos_tmp)
            print(betti_tmp[0][0])
#            far_tmp=pos_farest_dis(pos_tmp)
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            feature_train=feature_train+betti_tmp
            if label=='energy':
                target_train=target_train+energy_tmp
    if model_pick=='gbr':
        gbr=GBR(loss='ls',learning_rate=0.001,n_estimators=15000,max_depth=7,min_samples_split=5,subsample=1.0,max_features='sqrt',random_state=0)
    elif model_pick=='dense':
        gbr=make_dense_model()
#    print('kk')
    scaler_mm=MinMaxScaler()
    target_train=numpy.array(target_train)
    target_train=target_train.reshape(len(target_train),1)
#    target_train=scaler_mm.fit_transform(target_train)
#    target_train=target_train.tolist()
#    target_train=target_train[0]
    if split_ratio>0.001:
        feature_train1,feature_test1,target_train1,target_test1 =train_test_split(
        feature_train,target_train,test_size=0.3,random_state=0,shuffle=True)    
#        feature_test1,feature_test2,target_test1,target_test2 =train_test_split(
#        feature_test1,target_test1,test_size=0.5,random_state=0,shuffle=True)    
        if model_pick=='gbr':
            gbr.fit(feature_train1,target_train1)
        elif  model_pick=='dense':
            feature_train1=numpy.array(feature_train1)
            feature_test1=numpy.array(feature_test1)
            target_train1=target_train1.ravel()
            target_test1=target_test1.ravel()
            gbr.fit(feature_train1,target_train1, epochs=epo, batch_size=batc,validation_data=(feature_test1,target_test1),shuffle=True)
        pred_test=gbr.predict(feature_test1)
        RMSE=metrics.mean_squared_error(target_test1,pred_test)
        Pea=pearsonr(target_test1.flatten(),pred_test.flatten())
        
        pred_train=gbr.predict(feature_train1)
        print('train r2:%f'%metrics.r2_score(target_train1,pred_train))
        print('valid r2:%f'%metrics.r2_score(target_test1,pred_test))
        print('valid pearson:%f'%Pea[0])
        plt.figure(dpi=300, figsize=(5, 3))
        plt.scatter(target_test1,pred_test)
#        plt.savefig(('x'+str(far_x)+'valid.png'))
        plt.show()
        
#        pred_test2=gbr.predict(feature_test2)
#        RMSE2=metrics.mean_squared_error(target_test2,pred_test2)
#        Pea2=pearsonr(target_test2.flatten(),pred_test2.flatten())
#        print('test r2:%f'%metrics.r2_score(target_test2,pred_test2))
#        print('test pearson:%f'%Pea2[0])
#        plt.scatter(target_test2,pred_test2)
#        plt.savefig(('x'+str(far_x)+'test.png'))
#        plt.show()
        return feature_train1,feature_test1,target_train1,target_test1,gbr,RMSE,scaler_mm
    elif split_ratio<=0.001:
        feature_train1,target_train1 =shuffle(feature_train,target_train)
        feature_test1=[]
        target_test1=[]
        if model_pick=='gbr':
            gbr.fit(feature_train1,target_train1)
        elif  model_pick=='dense':
            feature_train1=numpy.array(feature_train1)
#            feature_test1=numpy.array(feature_test1)
            target_train1=target_train1.ravel()
#            target_test1=target_test1.ravel()
            gbr.fit(feature_train1,target_train1, epochs=epo, batch_size=batc,shuffle=True)
        RMSE='null'
        pred_train=gbr.predict(feature_train1)
        print('train:%f'%metrics.r2_score(target_train1,pred_train))
        return feature_train1,feature_test1,target_train1,target_test1,gbr,RMSE,scaler_mm
    
def get_compare_model(target,rank_list,model=0,scaler_mm=False): #feature_comp,target_comp = get_compare_model('20',['1'])
                                    #feature_comp,target_comp,pred_comp,RMSE_comp,pea= get_compare_model('20',['1'],model,scaler_mm)
    feature_comp=[]
    target_comp=[]
    target_file1=work_file+'/'+target
    os.chdir(target_file1)
    atom_num=int(target)
    label='energy'
    for j1 in rank_list:
        pos_tmp,energy_tmp,mag_tmp=analysis_vasp(j1,atom_num)
        betti_tmp=pos2betti(pos_tmp)
        far_tmp=pos_farest_dis(pos_tmp)
#        for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#            betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
        betti_tmp,energy_tmp,mag_tmp=betti_clean(betti_tmp,energy_tmp,mag_tmp)
        feature_comp=feature_comp+betti_tmp
        if label=='energy':
            energy_tmp=calc_bonding_energy(energy_tmp,atom_num,if_even=1)
            target_comp=target_comp+energy_tmp
    if scaler_mm:
        target_comp=numpy.array(target_comp)
        target_comp=target_comp.reshape(len(target_comp),1)
#        target_comp=scaler_mm.transform(target_comp)
    print(len(feature_comp))
    print(len(target_comp))
    if model==0:
        os.chdir(work_file)
        return feature_comp,target_comp
    else:
#        for i_f in range(len(feature_comp)):
#            feature_comp[i_f]=numpy.array(feature_comp[i_f])
        feature_comp=numpy.array(feature_comp)
        pred_comp=model.predict(feature_comp)
        RMSE=metrics.mean_squared_error(target_comp,pred_comp.tolist())
        Pea=pearsonr(target_comp.flatten(),pred_comp.flatten())
        print('r2:%f'%metrics.r2_score(target_comp,pred_comp))
        print('RMSE:%f'%RMSE)
        print('pearson:%f'%Pea[0])
        plt.figure(dpi=300, figsize=(5, 3))
        plt.plot(target_comp,pred_comp,'o',markersize=2)
        os.chdir(work_file)
        
        plt.savefig(('x'+str(far_x)+'comp.png'))
        plt.show()
        return feature_comp,target_comp,pred_comp,RMSE,Pea
def model_predict_pos(target,rank_iteration,model,pop_size,predf='pred'): #this def must start with rank as 0,target must be a str,iteration is a int,
    # model should have been trained,      model_predict_pos('7',10,model,50)
    os.chdir((work_file+'/'+target+predf))
    atom_num=int(target)
    record_file=work_file+'/'+target+predf+'/'+'pred.out'
    pos_all=[]
    ener_all=[]
    gbest_pos=[]
    gbest_ener=[]
    pso_w=pso_w1
    record_all=[]
    pos_all_all=[]
    for i5 in range(pop_size):
        pos_all.append([])
        ener_all.append([])
        record_all.append([])
        for i_re1 in range(atom_num):
            record_all[i5].append([])
    add_how_much=pop_size
    for i1 in range(10):
        file_calc=str(i1)
        if os.path.exists( file_calc):
            pass
        else:
            start_file=i1
            break
    for i2 in range(rank_iteration):
        if i2==round(rank_iteration*2/3):
            pso_w=pso_w1/2
#        if i2<5:
#            pso_r=0.5
#        else:
#            pso_r=0
        file_calc=start_file+i2
        os.mkdir(str(file_calc))
        posID_all=[]
        pos_itr_all=[]
        vw=[[]]
        for i9 in range(pop_size):
            vw[0].append([])
            for i10 in range(atom_num):
                vw[0][i9].append([0,0,0])
        if i2==0 and start_file==0: 
            add_how_much1=round(add_how_much/2)
            add_how_much2=add_how_much-add_how_much1
            pos_itr_all1,posID_all1=random_point_gen(atom_num,[],[],add_how_much1,'BCC')  
            pos_itr_all2,posID_all2=random_point_gen(atom_num,[],[],add_how_much2,'FCC')
            posID_all=posID_all1+posID_all2
            pos_itr_all=pos_itr_all1+pos_itr_all2
            for i in range(len(pos_itr_all)):
                pos_all_all.append(pos_itr_all[i])
#                pos_itr_all[i],e=pos_vibra(pos_itr_all[i],model)
            for i in range(1,len(pos_itr_all)+1):
                file_calc_1=(str(file_calc)+'/%d'%i)
                os.mkdir(file_calc_1)
                make_vasp(file_calc_1,pos_itr_all[i-1],if_run=False)
                ##predict
            betti_tmp=pos2betti(pos_itr_all)
            far_tmp=pos_farest_dis(pos_itr_all)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            pred_energy=model.predict(numpy.array(betti_tmp))
            pred_energy=pred_energy.tolist()
            best_pos_index=pred_energy.index(min(pred_energy))+1
            with open(record_file,'a') as rf:
                rf.write((str(file_calc)+'\n'))
                rf.write(str(best_pos_index)+'\n') 
                rf.write(str(min(pred_energy))+'\n') 
                rf.write(str(pos_itr_all[best_pos_index-1])+'\n') 
            for j3 in range(pop_size):
                ener_all[j3].append(pred_energy[j3])
                pos_all[j3].append(pos_itr_all[j3])
            mag_all=list(range(pop_size)) 
   
            pos2,energy2,mag2=pos_ener_sort(pos_itr_all,pred_energy,mag_all)
            gbest_pos.append(pos2[0])
            gbest_pos.append(pos2[1])
            gbest_pos.append(pos2[2])
            gbest_ener.append(energy2[0])
            gbest_ener.append(energy2[1])
            gbest_ener.append(energy2[2])

        else:
            if len(gbest_pos)==0:    #first genration are calculated , later are predicted
                pos1,energy1,mag1 = analysis_vasp('%d'%(file_calc-1),atom_num)
                pos2,energy2,mag2=pos_ener_sort(pos1,energy1,mag1)
                for j3 in range(pop_size):
                    ener_all[j3].append(energy1[j3])
                    pos_all[j3].append(pos1[j3])
                gbest_pos.append(pos1[best_pos_index-1])
                gbest_pos.append(pos2[0])
                gbest_pos.append(pos2[1])
                gbest_pos.append(pos2[2])
                gbest_ener.append(energy2[0])
                gbest_ener.append(energy2[1])
                gbest_ener.append(energy2[2])



            gbest_mag=list(range(len(gbest_pos)))
            gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
            

            gbest_fin_now=gbest_pos[-3]   
            gbest_fin_now.sort()
            if len(gbest_pos2)>=4:
                gbest_fin=gbest_pos2[random.randint(0,3)]
            else:
                gbest_fin=gbest_pos2[len(gbest_pos2)-1]
            gbest_fin.sort()
            vw.append([])
            for i3 in range(len(pos_all)):
                pbest_pos=[]
                pbest_ener=[]
                pbest_mag=[]
                vw[-1].append([])
                for i41 in range(len(pos_all[i3])):
                    pbest_pos.append(pos_all[i3][i41])
                    pbest_ener.append(ener_all[i3][i41])
                    pbest_mag.append(1)
                pbest_pos2,pbest_ener2,pbest_mag2=pos_ener_sort(pbest_pos,pbest_ener,pbest_mag)
                pbest_fin=pbest_pos2[0]
#                pbest_fin.sort()
                pos_now=pbest_fin#pbest_pos[-1]
                #pos_now.sort()
                pos_now,gbest_fin_now=search_close_route(pos_now,gbest_fin_now)
                pos_now,gbest_fin=search_close_route(pos_now,gbest_fin)
                go_on1=True
                pso_r=pso_r1
                try_lim=0
                while go_on1:
                    too_far_toomuch=0
                    too_close_toomuch=0
                    new_pos=[]
                    new_pos_w=[]
                    new_pos_c1=[]
                    new_pos_c2=[]
                    
                    for i4 in range(atom_num):

                        #new_pos_w.append(list(map(lambda x: 0*random.randint(-1,1)*random.random() ,pos_now[i4] )) )
#                        z_z=random.randint(-1,1)  
                        new_pos_c1.append(list(map(lambda x,y: (y-x)*pso_c1*random.random(),pos_now[i4],gbest_fin_now[i4] )) )
                        new_pos_c2.append(list(map(lambda x,y: (y-x)*pso_c2*random.random(),pos_now[i4],gbest_fin[i4] )) )
                        v_tmp=list(map(lambda x,y,z: pso_w*x+y+z,vw[-2][i3][i4],new_pos_c1[i4],new_pos_c2[i4]))
                        
                        if random.randint(0,1):
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i3][i4],new_pos_c2[i4]))
                        elif random.randint(0,1):
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i3][i4],new_pos_c1[i4]))
                        else:
                            v_tmp=list(map(lambda x,y,z: pso_w*x+y+z,vw[-2][i3][i4],new_pos_c2[i4],new_pos_c1[i4]))
                        vw[-1][i3].append(list(map(too_quick,v_tmp)))
                        new_pos.append(list(map(lambda y,ori: y+ori+pso_r*random.randint(-1,1)*random.random() ,vw[-1][i3][i4],pos_now[i4])))
                        record_all[i3][i4].append(pos_now[i4])
                        record_all[i3][i4].append(pbest_fin[i4])
                        record_all[i3][i4].append(gbest_fin[i4])
                        record_all[i3][i4].append(new_pos_c1[i4])
                        record_all[i3][i4].append(new_pos_c2[i4])
                    try_lim=try_lim+1
                    if try_lim== 10000:
                        new_pos=make_pos_fit(new_pos)
                        break
                    if too_far(new_pos,3.45):
                        print('too far')
                        go_on1=True
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                        too_far_toomuch=too_far_toomuch+1
                        if too_far_toomuch<10:
                            for j_del in range(len(vw[-1][i3])-1,-1,-1):
                                del vw[-1][i3][j_del]
                            continue
                        else:
                            new_pos1,posID_all1=random_point_gen(atom_num,[],[],1,'BCC')
                            new_pos=new_pos1[0]
                    if too_close(new_pos,2.45):
                        print('too close')
                        go_on1=True
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                        too_close_toomuch=too_close_toomuch+1
                        if too_close_toomuch<10:
                            for j_del in range(len(vw[-1][i3])-1,-1,-1):
                                del vw[-1][i3][j_del]
                            continue
                        else:
                            new_pos1,posID_all1=random_point_gen(atom_num,[],[],1,'BCC')
                            new_pos=new_pos1[0]
                            
                    if check_dup(new_pos,pos_now,(1/far_x)*0.75):
                        go_on1=True
                        for j_del in range(len(vw[-1][i3])-1,-1,-1):
                            del vw[-1][i3][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                        continue
#                    new_pos=make_pos_fit(new_pos)
                    break
                    if check_dup2(pos_all_all,new_pos,(1/far_x)*0.5):
                        go_on1=True
                        for j_del in range(len(vw[-1][i3])-1,-1,-1):
                            del vw[-1][i3][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.02
                        continue
#                    new_pos=make_pos_fit(new_pos)
                    break
                pos_all_all.append(new_pos)
#                new_pos,e=pos_vibra(new_pos,model)
                pos_itr_all.append(new_pos)
            for i7 in range(1,len(pos_itr_all)+1):
                file_calc_1=(str(file_calc)+'/%d'%i7)
                os.mkdir(file_calc_1)
                make_vasp(file_calc_1,pos_itr_all[i7-1],if_run=False)
            betti_tmp=pos2betti(pos_itr_all)
            far_tmp=pos_farest_dis(pos_itr_all)
#            for j3 in range(len(betti_tmp)):
#                betti_tmp[j3].append(round(far_tmp[j3],2))
#                betti_tmp[j3].append(round(far_tmp[j3]*far_k)/far_k)
            pred_energy=model.predict(numpy.array(betti_tmp))
            pred_energy=pred_energy.tolist()
            best_pos_index=pred_energy.index(min(pred_energy))+1

            with open(record_file,'a') as rf:
                rf.write((str(file_calc)+'\n'))
                rf.write(str(best_pos_index)+'\n')  
                rf.write(str(min(pred_energy))+'\n') 
                rf.write(str(pos_itr_all[best_pos_index-1])+'\n') 
            for j3 in range(pop_size):
                ener_all[j3].append(pred_energy[j3])
                pos_all[j3].append(pos_itr_all[j3])
            mag_all=list(range(pop_size))  
            pos_21,energy_21,mag_21=pos_ener_sort(pos_itr_all,pred_energy,mag_all)
            gbest_pos.append(pos_21[0])
            gbest_pos.append(pos_21[1])
            gbest_pos.append(pos_21[2])
            gbest_ener.append(energy_21[0])
            gbest_ener.append(energy_21[1])
            gbest_ener.append(energy_21[2])
        os.chdir((work_file+'/'+target+predf))
    gbest_mag=list(range(len(gbest_pos)))
    gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
    already_cal=[]
    for best_i in range(1,len(gbest_pos2)):#len(gbest_pos2)+1
        pwd_file=os.getcwd()
        if check_dup2(already_cal,gbest_pos2[best_i-1],4/far_x):
            continue
        already_cal.append(gbest_pos2[best_i-1])
        file_calc_best=pwd_file+'/final'+str(best_i)
        os.mkdir(file_calc_best)
        make_vasp(file_calc_best,gbest_pos2[best_i-1],False,True,if_pred=True)
        os.chdir(pwd_file)

#        with open('final%d.gjf'%(best_i-1),'w') as fg:
#            fg.write('%chk=C.chk\n%mem=10GB\n\n# bpw91/6-31G pop=full nosym scf=(maxcycle=80,xqc)\n\npso\n\n')
#            fg.write('0 %d\n'%gbest_mag2[best_i-1])
#            for item in gbest_pos2[best_i]:
#                fg.write('Li %15.8f %15.8f %15.8f\n' % tuple(item))
#            fg.write('\n')  
    return already_cal
        
def pos2standard_coord(pos):  #pos list should be assgined here,we choose mass center point as center point , and the longgert distance 
    new_pos=[]                            # as x axis
    for i_1 in range(len(pos)):
        pos_tmp=[]
        x_sum=0
        y_sum=0
        z_sum=0
        for j_1 in range(len(pos[i_1])):
            x_sum=x_sum+pos[i_1][j_1][0]
            y_sum=y_sum+pos[i_1][j_1][1]
            z_sum=z_sum+pos[i_1][j_1][2]
        center=[x_sum/len(pos[i_1]),y_sum/len(pos[i_1]),z_sum/len(pos[i_1])]
        for j_1 in range(len(pos[i_1])):
            pos_tmp.append([pos[i_1][j_1][0]-center[0] , pos[i_1][j_1][1]-center[1] , pos[i_1][j_1][2]-center[2]])
        new_pos.append(pos_tmp)
    return new_pos,center

def search_close_route(pos1,pos2):
    pos_tmp=numpy.array(pos2)
    pos2=pos_tmp.tolist()
    pos_tmp=numpy.array(pos1)
    pos1=pos_tmp.tolist()
    #pos1.sort()
    pos2.sort()
    new_pos2=[]
    #print(pos1)
    #print(pos2)
    for ii_1 in range(len(pos1)):
        dist_mat=[]
        for ii_2 in range(len(pos2)):
            temp=math.pow(pos1[ii_1][0]-pos2[ii_2][0],2)+math.pow(pos1[ii_1][1]-pos2[ii_2][1],2)+math.pow(pos1[ii_1][2]-pos2[ii_2][2],2)
            dist_mat.append(temp)
        no_temp_close=dist_mat.index(min(dist_mat))
        new_pos2.append(pos2[no_temp_close])
        del pos2[no_temp_close]
    return pos1,new_pos2
def make_pos_fit(pos):
    pos2,c1=pos2standard_coord([pos])
    pos1=pos2[0]
    max_dis0=math.pow(atom_distances-0.2,2)
    max_dis1=math.pow(atom_distances+0.4,2)
    max_dis2=math.pow(atom_distances+0.8,2)
    iter_1=0
#    for iter_1 in range(len(pos1)*2):
    while True:
        iter_1= iter_1+1
        change_sum=0
        for i_11 in range(len(pos1)-1):
            i_1=random.randint(0,len(pos1)-2)
            for j_1 in range(i_1+1,len(pos1)):
                bef_dis=math.pow(pos1[i_1][0]-pos1[j_1][0],2) + math.pow(pos1[i_1][1]-pos1[j_1][1],2) +math.pow(pos1[i_1][2]-pos1[j_1][2],2)
                if bef_dis > max_dis0 and bef_dis < max_dis1+0.1 :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                           
                elif bef_dis > max_dis1 and bef_dis < max_dis2 :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances+0.4,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
                elif bef_dis < max_dis0  :
                    ltr_dis=math.sqrt(bef_dis/math.pow(atom_distances-0.3,2))
                    bef_x=pos1[j_1][0]-pos1[i_1][0]
                    bef_y=pos1[j_1][1]-pos1[i_1][1]
                    bef_z=pos1[j_1][2]-pos1[i_1][2]
                    ltr_x=(bef_x/ltr_dis+bef_x*3)/4
                    ltr_y=(bef_y/ltr_dis+bef_y*3)/4
                    ltr_z=(bef_z/ltr_dis+bef_z*3)/4
                    pos1[j_1]=[pos1[i_1][0]+ltr_x,pos1[i_1][1]+ltr_y,pos1[i_1][2]+ltr_z]
                    change_sum=change_sum+abs(ltr_x-bef_x)+abs(ltr_y-bef_y)+abs(ltr_y-bef_y)
        if change_sum<0.005 or iter_1>50:
            break                    
    return pos1

def continue_pos(atom_num,cal_method,if_run,iteration,start_po=0,model=False):   #iteration ==== the final dir number  
    record_file=work_file+'/'+'%d'%atom_num+'/'+'pred.out'

    pos_all=[]
    pos_cal_all_all=[]
    ener_all=[]
    gbest_pos=[]
    gbest_ener=[]
    os.chdir((work_file+'/'+'%d'%atom_num))
    pop_list=os.listdir('0')
    pop_size=len(pop_list)
    vw=[[]]
    pso_w=pso_w1
    for i9 in range(pop_size):
        vw[0].append([])
        for i10 in range(atom_num):
            vw[0][i9].append([0,0,0])
    for i5 in range(pop_size):
        pos_all.append([])
        ener_all.append([])
    for it in range(start_po,iteration+start_po+1):
        time_use=0
        if it==round(iteration*2/3):
            pso_w=pso_w1/3
        while True:
            wether_cal2=if_cal_done_vasp('%d'%it,pop_size)
            if cal_method=='vasp' and wether_cal2:
                pos1,energy1,mag1 = analysis_vasp('%d'%it,atom_num)
                best_pos_index=energy1.index(min(energy1))
                for k_1 in range(len(pos1)):
                    pos_cal_all_all.append(pos1[k_1])
                print('best pos : %d'%best_pos_index )
                with open(record_file,'a') as rf:
                    rf.write((str(it)+'\n'))
                    for i_xx in pos1[best_pos_index]:
                        test_str = " ".join([str(i_xxx) for i_xxx in i_xx])
                        rf.write(test_str+'\n') 
                    rf.write(str(min(energy1))+'\n') 
                    rf.write(str(pos1[best_pos_index])+'\n') 
                while len(pos1)<pop_size:
                    pos1.append(pos1[0])
                    energy1.append(energy1[0])
                    mag1.append(mag1[0])
                for i6 in range(pop_size):
                    pos_all[i6].append(pos1[i6])
                    ener_all[i6].append(energy1[i6])
                pos2,energy2,mag2=pos_ener_sort(pos1,energy1,mag1)
                gbest_pos.append(pos2[0])
                gbest_pos.append(pos2[1])
                gbest_pos.append(pos2[2])
                gbest_ener.append(energy2[0])
                gbest_ener.append(energy2[1])
                gbest_ener.append(energy2[2])
                break
            wether_cal=if_cal_done_gaus('%d'%it,pop_size)
            if cal_method=='gaus' and wether_cal:
                pos1,energy1,mag1 = analysis_vasp('%d'%it,atom_num)
                best_pos_index=energy1.index(min(energy1))+1
                for k_1 in range(len(pos1)):
                    pos_cal_all_all.append(pos1[k_1])
                print('best pos : %d'%best_pos_index )
                with open(record_file,'a') as rf:
                    rf.write((str(it)+'\n'))
                    for i_xx in pos1[best_pos_index]:
                        test_str = " ".join([str(i_xxx) for i_xxx in i_xx])
                        rf.write(test_str+'\n') 
                    rf.write(str(min(energy1))+'\n')  
                while len(pos1)<pop_size:
                    pos1.append(pos1[0])
                    energy1.append(energy1[0])
                    mag1.append(mag1[0])
                for i6 in range(pop_size):
                    pos_all[i6].append(pos1[i6])
                    ener_all[i6].append(energy1[i6])
                pos2,energy2,mag2=pos_ener_sort(pos1,energy1,mag1)
                gbest_pos.append(pos2[0])
                gbest_pos.append(pos2[1])
                gbest_pos.append(pos2[2])
                gbest_ener.append(energy2[0])
                gbest_ener.append(energy2[1])
                gbest_ener.append(energy2[2])
                break
            else:
                print('calculation not done yet!! good luck !')
                time.sleep(5)

#                if_cal_done_vasp_re('%d'%it,pop_size)
            
#                for i1 in range(len(pos1)-1):
#                    for i2 in range(1,len(pos1)-i1-1):
#                        if check_dup(pos2[i1],pos2[i1+i2]):
#                            del pos2[i1+i2];del energy2[i1+i2];del mag2[i1+i2]                            
#                if len(pos2)<pop_size:
#                    pos_all3=[]
#                    posID_all3=[]
#                    pos_all3,posID_all3=random_point_gen(atom_num,pos2,posID_all,pop_size-len(pos2))
                
            #for i in range(pop_size):     

        if not os.path.exists('%d'%(it+1)) and it != iteration+start_po:
            gbest_mag=[]
            for i1 in range(len(gbest_pos)):
                gbest_mag.append(1)
            
            gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
            gbest_fin_now=gbest_pos[-3]   
            gbest_fin_now.sort()

            gbest_fin=gbest_pos2[random.randint(0,len(gbest_pos2)-1)]#len(gbest_pos2)-1
     
            gbest_fin.sort()
            pos_cal_all=[]
        
            vw.append([])
            for i2 in range(len(pos_all)):
                pbest_pos=[]
                pbest_ener=[]
                pbest_mag=[]
                vw[-1].append([])
                for i3 in range(len(pos_all[i2])):
                    pbest_pos.append(pos_all[i2][i3])
                    pbest_ener.append(ener_all[i2][i3])
                    pbest_mag.append(1)
                pbest_pos2,pbest_ener2,pbest_mag2=pos_ener_sort(pbest_pos,pbest_ener,pbest_mag)
               # print(pos_all)
               # print(pbest_pos)
                pbest_fin=pbest_pos2[0]
                pbest_fin.sort()
                pos_now=pbest_pos[-1]
                #pos_now.sort()
               # print(pbest_fin)
                pos_now,pbest_fin=search_close_route(pos_now,pbest_fin)
                pos_now,gbest_fin=search_close_route(pos_now,gbest_fin)
                pos_now,gbest_fin_now=search_close_route(pos_now,gbest_fin_now)
#                if i2==best_pos_index-1:
                #print(pbest_fin)
#                    print(pos_now)
                go_on1=True
                pso_r=pso_r1
                too_far_toomany=0
                dup_all_toomany=0                
                too_close_toomany=0
                try_lim=0
                while go_on1:
                    new_pos=[]
                    new_pos_w=[]
                    new_pos_c1=[]
                    new_pos_c2=[]
                    try_lim=try_lim+1
                    for i4 in range(atom_num):
                        
#                        z_z=random.randint(-1,1)   
                        #new_pos_w.append(list(map(lambda x: x*pso_w*random.randint(-1,1) ,pos_now[i4] )) )
                        new_pos_c1.append(list(map(lambda x,y: (y-x)*pso_c1*random.random() ,pos_now[i4],pbest_fin[i4] )) )#gbest_fin_now[i4]
                        new_pos_c2.append(list(map(lambda x,y: (y-x)*pso_c2*random.random(),pos_now[i4],gbest_fin[i4] )) )
                        if random.randint(0,1):
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i2][i4],new_pos_c1[i4]))
                        else:
                            v_tmp=list(map(lambda x,y: pso_w*x+y,vw[-2][i2][i4],new_pos_c2[i4]))

                        vw[-1][i2].append(list(map(too_quick,v_tmp)))
                        new_pos.append(list(map(lambda y,ori: y+ori+pso_r*random.randint(-1,1)*random.random() ,
                                                vw[-1][i2][i4],pos_now[i4])))
                       # print('%f %f %f\n'%(vw[-2][i2][i4][1],new_pos_c1[i4][1],new_pos_c2[i4][1]))
                       
                    if try_lim== 10000:
                        new_pos=make_pos_fit(new_pos)
                        break
                    if too_far(new_pos):
                        go_on1=True
                        print(pso_r)
                        if pso_r<0.6:
                            pso_r=pso_r+0.05

                        print('too far')
                        too_far_toomany=too_far_toomany+1

                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        continue

                    if too_close(new_pos):
                        go_on1=True
                        if pso_r<0.6:
                            pso_r=pso_r+0.05
                        print('too close')
                        too_close_toomany=too_close_toomany+1

                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        continue

                    if check_dup(new_pos,pos_now):
                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.05
                        print('dup now')
                        continue
#                    new_pos=make_pos_fit(new_pos)
#                    if model:
#                        new_pos,ee=pos_vibra(new_pos,model)   
                    if check_dup2(pos_cal_all,new_pos):
                        go_on1=True
                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.05
                        print('dup cal all')
                        continue
                    if check_dup2(gbest_pos,new_pos):
                        go_on1=True
                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        if pso_r<0.6:
                            pso_r=pso_r+0.05
                        print('dup best all')
                        continue
                    if check_dup2(pos_cal_all_all,new_pos):
                        go_on1=True
                        if pso_r<0.6:
                            pso_r=pso_r+0.05
                        print('too close')
                        dup_all_toomany=dup_all_toomany+1
                        for j_del in range(len(vw[-1][i2])-1,-1,-1):
                            del vw[-1][i2][j_del]
                        continue


                    break
                    
                
                pos_cal_all.append(new_pos)
            file_calc='%d'%(it+1)
            os.mkdir(file_calc)
            if len(pos_cal_all)!=pop_size:
                print('pop_size problem')
            for i in range(1,len(pos_cal_all)+1):
                file_calc_1=(file_calc+'/%d'%i)
                os.mkdir(file_calc_1)
                make_vasp(file_calc_1,pos_cal_all[i-1],if_run,if_opt=True)
                check_if_sleep()
    os.chdir((work_file+'/'+'%d'%atom_num))
    gbest_mag=list(range(len(gbest_pos)))
    gbest_pos2,gbest_ener2,gbest_mag2=pos_ener_sort(gbest_pos,gbest_ener,gbest_mag)
    with open('final%d.gjf'%it,'w') as fg:
        fg.write('%chk=C.chk\n%mem=10GB\n\n# bpw91/6-31G pop=full nosym opt=(maxstep=20,maxcycle=10,loose) scf=(maxcycle=80,xqc)\n\npso\n\n')
        fg.write('0 %d\n'%gbest_mag2[0])
        for item in gbest_pos2[0]:
            fg.write('Li %15.8f %15.8f %15.8f\n' % tuple(item))
        fg.write('\n')
    os.chdir(work_file)
def make_dense_model():
    model=Sequential()
    #model.add(Convolution1D(128,3,border_mode='same',input_shape=(3,int(vector_length/3),)))
    #model.add(Activation('relu'))
    #model.add(Convolution1D(128,3))
    #model.add(Activation('relu'))
    #model.add(AveragePooling1D(pool_length=2,stride=None,border_mode='valid'))
    #model.add(Dropout(0.25))
    #model.add(Convolution1D(256,3,border_mode='same'))
    #model.add(Activation('relu'))
    #model.add(Convolution1D(256,3))
    #model.add(Activation('relu'))
    #model.add(AveragePooling1D(pool_length=2,stride=None,border_mode='valid'))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    model.add(BatchNormalization())
    for i in range(dense_layers):
        model.add(Dense(dense_nodes))
        model.add(Activation('tanh'))
    model.add(Dense(dense_nodes))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))
    adam=Adam(lr=learn_r)
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_absolute_error'])
    return model
def check_if_sleep(howlong=10):
    while True:
        qstat=subprocess.check_output('qstat',universal_newlines=True)
        qstat=qstat.split('\n')
        if len(qstat)>65:
            time.sleep(howlong)
        else:
            break

global work_file
global sole_energy
global pso_w1
global pso_r1
global pso_c1
global pso_c2
global atom_distances
global far_x
global dense_nodes
global dense_layers
global learn_r
#far_k=100
pso_r1=0
pso_w1=0.9
pso_c1=2
atom_distances=3
pso_c2=2
sole_energy=-.29378687E+00
work_file='/udata/chenx/topology/VASP12'
os.chdir(work_file)
far_x=10
#time.sleep(25000)
dense_nodes=800
dense_layers=2
learn_r=0.0001
#for i in ['NN','GBR']:
feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','20'],0.3,'GBR',epo=200,batc=128,plus=False)
#save_feature(feature_train,target_train,feature_test,target_test,name='feature_target_3_10_x_10_j_20')
#node_record=''
#for dense_nodes in [600,800,1200]:
#    for dense_layers in [1,2,3]:
#        for batch_i in [64,96,128]:
#            feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','20'],0.3,'NN',epo=100,batc=batch_i,plus=False)
#            node_record=node_record+'nodes: '+str(dense_nodes)+'layers: '+str(dense_layers)+' ' +str(batch_i)+':'+'   rmse '+str(RMSE)+'\n'
#            print(node_record)
#for batch_i in [50,200]:
#feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=load_feature(name='new_feature_target_3_10_x_3_j_20',split_ratio=0.3,epo=5,batc=50)
#feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=load_feature(name='feature_target_3_10_x_10_j_20',split_ratio=0.3,epo=2,batc=50)
#feature_comp0,target_comp0,pred_comp0,RMSE_comp,pea = get_compare_model('10',['2'],model,scaler_mm)
#feature_comp1,target_comp1,pred_comp1,RMSE_comp,pea = get_compare_model('20',['2'],model,scaler_mm)
#feature_comp2,target_comp2,pred_comp2,RMSE_comp,pea = get_compare_model('40',['2'],model,scaler_mm)
#feature_comp3,target_comp3,pred_comp3,RMSE_comp,pea = get_compare_model('15',['2'],model,scaler_mm)


#plt.figure(figsize=(3, 3))
#plt.xlim(-1.3,-0.1)
#plt.ylim(-1.3,-0.1)
#
##plt.plot(target_comp0,pred_comp0,'yo',markersize=2)
#plt.plot(target_comp1,pred_comp1,'ro',markersize=2)
#plt.plot(target_comp2,pred_comp2,'bo',markersize=2)
##plt.plot(target_comp3,pred_comp3,'go',markersize=2)
#plt.show()
#                node_record=node_record+'nodes: '+str(dense_nodes)+'layers: '+str(dense_layers)+' ' +str(epo)+':'+'pea '+str(pea[0])+'   rmse '+str(RMSE_comp)+'\n'
##  x=10 nodes: 800      layers: 4    epo: 5:    batch: 50        pea 0.9884036567048737   rmse 1.0356497988976052
##  x=3 nodes: 1200      layers: 3   epo: 5:    batch: 50        pea 0.9824036567048737   rmse 0.9356497988976052
#            print((dense_layers))
#            print(dense_nodes)
#            print(batch_i)
#    history=model.fit(feature_train,target_train, epochs=10, batch_size=200,validation_data=(feature_test,target_test),shuffle=True)
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','20'],0.0)
#save_feature(feature_train,target_train,feature_test,target_test,name='new_feature_target_3_10_x_3_j_20')
#node_record=''
##for nodes in range(200,500,20):
###    feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','10'],0.3)
##    feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=load_feature(name='feature_target_3_10_x_10_j_10',split_ratio=0.3)
#feature_comp,target_comp,pred_comp,RMSE_comp,pea = get_compare_model('20',['2'],model,scaler_mm)
#    node_record=node_record+str(nodes)+':'+'pea '+str(pea[0])+'   rmse '+str(RMSE_comp)+'\n'
#for epo in [3]:
#    for batc in [20,40,50,60,70]:
#        feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=load_feature(name='new_feature_target_3_10_x_10_j_20',split_ratio=0.3,epo=epo,batc=batc)
#        feature_comp,target_comp,pred_comp,RMSE_comp,pea = get_compare_model('20',['2'],model,scaler_mm)
##    feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','10'],0.3)
#        feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=load_feature(name='feature_target_3_10_x_3_j_20',split_ratio=0.3,epo=epo,batc=batc)
#        feature_comp,target_comp,pred_comp,RMSE_comp,pea = get_compare_model('20',['2'],model,scaler_mm)
#        node_record=node_record+str(epo)+' '+str(batc)+':'+'pea '+str(pea[0])+'   rmse '+str(RMSE_comp)+'\n'
###save_feature(feature_train,target_train,feature_test,target_test,name='feature_target1_10')
###for apq in [5,8,10,12,15,18]:
###    apq=str(apq)
###    feature_train,feature_test,target_train,target_test,model1,RMSE=get_train_model(['7'],['0',apq],0.3)
#joblib.dump(model, '''train_3_10_even_model_x_2_j_21.m''')
#feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3','4','5','6','7','8','9','10'],['0','20'],0.3)
#feature_train,feature_test,target_train,target_test,model,RMSE,scaler_mm=get_train_model(['3'],['0','1'],0.3)
#feature_comp,target_comp,pred_comp,RMSE_comp = get_compare_model('11',['1'],model,scaler_mm)
#feature_comp,target_comp,pred_comp,RMSE_comp = get_compare_model('12',['1'],model,scaler_mm)

#feature_train,feature_test,target_train,target_test,model,RMSE=get_train_model(['3'],['0','6'],0.3)
##print('9')
#model=load_model('dense_3_10_x_10_j_10_epo_50_bat_200')
#xxxx=model_predict_pos('20',5000,model,50,'pred6')
#random_point_gen(atom_num,pos_all,posID_all,add_how_much)          
#pc=0.6
#pm=0.4
#
#feature_train1,target_train1,model=self_learn(['3'],6,mingeneration1=15,mingeneration2=5)
#    feature_train2,target_train2,model=self_learn(['4'],20,mingeneration1=25,mingeneration2=5)
#    feature_train3,target_train3,model=self_learn(['5'],50,mingeneration1=35,mingeneration2=5)
#    feature_train4,target_train4,model=self_learn(['6'],50,mingeneration1=35,mingeneration2=5)
#    feature_train5,target_train5,model=self_learn(['7'],50,mingeneration1=35,mingeneration2=5)
#    feature_train6,target_train6,model=self_learn(['8'],50,mingeneration1=35,mingeneration2=5)    
#    feature_train7,target_train7,model=self_learn(['9'],50,mingeneration1=49,mingeneration2=1)    
#    feature_train8,target_train8,model=self_learn(['10'],50,mingenerati on1=35,mingeneration2=5)    
#    feature_train9,target_train9,model=self_learn(['11'],50,mingeneration1=35,mingeneration2=5)    
#    ##  
#    atomnum=2
#    os.chdir('/udata/chenx/topology/VASP11/%d'%atomnum) 
#    ini_pos(atomnum,1,'vasp',True)                                 
#    continue_pos(atomnum,'vasp',True,3,0)   
#    atomnum=3
#    os.chdir('/udata/chenx/topology/VASP11/%d'%atomnum) 
#    ini_pos(atomnum,6,'vasp',True)                                 
#    continue_pos(atomnum,'vasp',True,30,0)   
#atomnum=3
#os.chdir('/udata/chenx/topology/VASP11/%d'%atomnum) 
#ini_pos(atomnum,6,'vasp',True)                                 
#continue_pos(atomnum,'vasp',True,20,0) 
#for pbs in range(40,4441):
#    atomnum=pbs
#    os.chdir('/udata/chenx/topology/VASP12/%d'%atomnum) 
#    ini_pos(atomnum,50,'vasp',True)                                 
#    continue_pos(atomnum,'vasp',True,20,0) 
#    atomnum=pbs
    
#    os.chdir('/udata/chenx/topology/VASP12/%d'%atomnum) 
#    ini_pos(atomnum,50,'vasp',True)                                 
#    continue_pos(atomnum,'vasp',True,30,0) 
