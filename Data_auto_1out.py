'''
Instruction: input format: python3 Total.py variable_name, split_percentage
            variable_name choose from {Effector,SkelPort,Element,Species,LPort,SzCl}
Example: python3 Total.py newestData.xlsx Element 0.3 T
'''
import sys
import itertools
import pandas as pd
import scipy.io
import math
import numpy as np
import random
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
#from imblearn.over_sampling import SMOTE

def Loaddata(xls_name, variable_name):
    df = pd.read_excel(xls_name,header=None)
    df=np.array(df)
    df=df.tolist()
    keys=[]
    ###########construct break edges data#################
    for i in range(0,len(df[0])):
        if df[0][i] == 'P/A_NotchA_Frag':
            PA_NotchA_Frag=i
        if df[0][i] == 'P/A_NotchC_Frag':
            PA_NotchC_Frag=i
        if df[0][i] == 'P/A_NotchD_Frag':
            PA_NotchD_Frag=i
        if df[0][i] == 'Notches':
            Notches=i
        if df[0][i] == 'Epiphysis':
            Epiphysis=i
        if df[0][i] == 'MaxDim':
            MaxDim=i
        if df[0][i] == 'MLIntervals':
            MLIntervals=i
        if df[0][i] == 'BkQty':
            BkQty=i
        if df[0][i] == 'SpecLabel':
            SpecLabel=i
        if df[0][i] == 'Use4ML':
            Use4ML=i
    ####Variables#########
        if df[0][i] == 'Effector':
            Effector=i
        if df[0][i] == 'ActorGroup':
            ActorGroup=i
        if df[0][i] == 'SkelPort':
            Skelport=i
        if df[0][i] == 'Element':
            Element=i
        if df[0][i]== 'Species':
            Species=i
        if df[0][i]=='SzCl':
            SzCl=i
        if df[0][i]=='LPort':
            LPort=i
    #####break curves#########
        if df[0][i] ==  'BkPlane1':
             BkPlane1=i
        if df[0][i] ==  'Gon1':
             Gon1=i
        if df[0][i] ==  'VMFracAng1':
             VMFracAng1=i
        if df[0][i] ==  'BkLmax1':
             BkLmax1=i
    ####
        if df[0][i] ==  'BkPlane2':
             BkPlane2=i
        if df[0][i] ==  'Gon2':
             Gon2=i
        if df[0][i] ==  'VMFracAng2':
             VMFracAng2=i
        if df[0][i] ==  'BkLmax2':
             BkLmax2=i
    ####
        if df[0][i] ==  'BkPlane3':
             BkPlane3=i
        if df[0][i] ==  'BkLmax3':
             BkLmax3=i
        if df[0][i] ==  'Gon3':
             Gon3=i
        if df[0][i] ==  'VMFracAng3':
             VMFracAng3=i
    ####
        if df[0][i] ==  'BkPlane4':
             BkPlane4=i
        if df[0][i] ==  'BkLmax4':
             BkLmax4=i
        if df[0][i] ==  'Gon4':
             Gon4=i
        if df[0][i] ==  'VMFracAng4':
             VMFracAng4=i
    ####
        if df[0][i] ==  'BkPlane5':
             BkPlane5=i
        if df[0][i] ==  'BkLmax5':
             BkLmax5=i
        if df[0][i] ==  'Gon5':
             Gon5=i
        if df[0][i] ==  'VMFracAng5':
             VMFracAng5=i
    ####
        if df[0][i] ==  'BkPlane6':
             BkPlane6=i
        if df[0][i] ==  'BkLmax6':
             BkLmax6=i
        if df[0][i] ==  'Gon6':
             Gon6=i
        if df[0][i] ==  'VMFracAng6':
             VMFracAng6=i
    ####
        if df[0][i] ==  'BkPlane7':
             BkPlane7=i
        if df[0][i] ==  'BkLmax7':
             BkLmax7=i
        if df[0][i] ==  'Gon7':
             Gon7=i
        if df[0][i] ==  'VMFracAng7':
             VMFracAng7=i
    ####
        if df[0][i] ==  'BkPlane8':
             BkPlane8=i
        if df[0][i] ==  'BkLmax8':
             BkLmax8=i
        if df[0][i] ==  'Gon8':
             Gon8=i
        if df[0][i] ==  'VMFracAng8':
             VMFracAng8=i
    #specimen=[]
    #allbreak=[]
    Variables={'Effector':Effector,'SkelPort':Skelport,'Element':Element,'Species':Species,'SzCl':SzCl,'LPort':LPort,'ActorGroup':ActorGroup}
    variable=Variables.get(variable_name)
    Fragments=[]
    for eachrow in df[1:]:
        countGon=0
        if eachrow[Use4ML]=='yes' and (not pd.isnull(eachrow[variable])):
            fragments=[]
            if not pd.isnull(eachrow[Gon1]):
                countGon+=1
            if not pd.isnull(eachrow[Gon2]):
                countGon+=1
            if not pd.isnull(eachrow[Gon3]):
                countGon+=1
            if not pd.isnull(eachrow[Gon4]):
                countGon+=1
            if not pd.isnull(eachrow[Gon5]):
                countGon+=1
            if not pd.isnull(eachrow[Gon6]):
                countGon+=1
            if not pd.isnull(eachrow[Gon7]):
                countGon+=1
            if not pd.isnull(eachrow[Gon8]):
                countGon+=1
          
            BrkQty=eachrow[BkQty]-countGon

            if not pd.isnull(eachrow[Gon1]):
                break1=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane1],
                         eachrow[Gon1],eachrow[VMFracAng1],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break1]
                fragments+=[break1]

            if not pd.isnull(eachrow[Gon2]):
                break2=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane2],
                         eachrow[Gon2],eachrow[VMFracAng2],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break2]
                fragments+=[break2]
                
            if not pd.isnull(eachrow[Gon3]):
                break3=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane3],
                         eachrow[Gon3],eachrow[VMFracAng3],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break3]
                fragments+=[break3]
         
            if not pd.isnull(eachrow[Gon4]):
                break4=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane4],
                         eachrow[Gon4],eachrow[VMFracAng4],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break4]
                fragments+=[break4]
          
            if not pd.isnull(eachrow[Gon5]):
                break5=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane5],
                         eachrow[Gon5],eachrow[VMFracAng5],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break5]
                fragments+=[break5]

            if not pd.isnull(eachrow[Gon6]):
                break6=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane6],
                         eachrow[Gon6],eachrow[VMFracAng6],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break6]
                fragments+=[break6]

            if not pd.isnull(eachrow[Gon7]):
                break7=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane7],
                         eachrow[Gon7],eachrow[VMFracAng7],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break7]
                fragments+=[break7]
             
            if not pd.isnull(eachrow[Gon8]):
                break8=[eachrow[MaxDim],eachrow[MLIntervals],eachrow[Epiphysis],BrkQty,eachrow[BkPlane8],
                         eachrow[Gon8],eachrow[VMFracAng8],eachrow[Notches],eachrow[PA_NotchA_Frag],
                        eachrow[PA_NotchC_Frag],eachrow[PA_NotchD_Frag],eachrow[SpecLabel],eachrow[variable]]
                #allbreak+=[break8]
                fragments+=[break8]
            
            if fragments!=[]:
                Fragments+=[[fragments,eachrow[variable]]]
                keys+=[[eachrow[SpecLabel],eachrow[variable]]]
    if variable_name=='Element' or 'LPort':
        for key in keys:
            if key[-1]=='other':
                keys.remove(key)
    
    #return allbreak
    return Fragments,keys

def leaveoneout(allbreak,keys,balance):
    Result=[]
    for key in keys:
        test=[]
        train=deepcopy(allbreak)
        for curve in allbreak:
            
            if curve[0][0][-2]==key[0]:
                test+=[[curve]]
                train.remove(curve)
              
        test=list(itertools.chain.from_iterable(test))

    
        train,classtype1=balancing_converting(train,balance)
        test,classtype2=balancing_converting(test,balance)
        #print(test[0:3])
        test=list(itertools.chain.from_iterable(test))
        train=list(itertools.chain.from_iterable(train))

        train=numerical_Label(train,classtype1)
        
        test=numerical_Label(test,classtype1)
        #print(test[0:3])
        xtrain,xtest=leaveoneout_split(train)
        ytrain,ytest=leaveoneout_split(test)
        #print(ytrain)
        clf=train_classifier(xtrain,xtest)
        
        prediction=clf.predict(ytrain).tolist()
        Result+=[[most_frequent(prediction),ytest[0]]]
        #print([most_frequent(prediction),ytest[0]])
    right=0
    total=0
    for i in Result:
        if i[0]==i[1]:
            right+=1
            total+=1
        else:
            total+=1
    print("correct:",right)
    print("Incorrect:",total-right)
    print(right/total)



def leaveoneout_split(breaks):
    train=[]
    test=[]
    for curve in breaks:
        train+=[curve[0:-1]]
        test+=[curve[-1]]
    return train,test
    


def balancing_converting(allbreak,balance):
    allbreak=deepcopy(allbreak)
    classtype=[]
    for i in range(0,len(allbreak)):
        if str(allbreak[i][-1]) not in classtype:
            classtype+=[str(allbreak[i][-1])]
    
    classamount=[0]*len(classtype)
    Classes=[]
    for j in range(0,len(classtype)):
        classes=[]
        for i in range(0,len(allbreak)):
            if str(allbreak[i][-1])==classtype[j]:
                classamount[j]+=1
                classes+=allbreak[i][:-1]
        Classes+=[classes]


    #for i in range(0,len(classtype)):
    #    print('Class ',classtype[i],'has ',classamount[i],'fragments.')
    
    if balance==True:
        for i in range(0,len(classamount)):
            if classamount[i]==min(classamount) and classamount[i]!=0:
                baseclass=i
                
        basenumber=classamount[baseclass]
        
        for i in [x for x in range(0,len(classtype)) if classamount[x]>basenumber]:
            I=np.random.choice(np.arange(classamount[i]),size=basenumber,replace=False)
            Classes[i]=[Classes[i][j] for j in I]

        allbreak=[]
        print('###########################################################')
        for i in [x for x in range(0,len(classtype)) if classamount[x]>basenumber]:
            allbreak+=Classes[i]
            print('Class ',classtype[i],'added, with ',basenumber,' fragments')
        allbreak+=Classes[baseclass]
        print('Class ',classtype[baseclass],'added, with ',basenumber,' fragments')
        print('Total number of fragments for model training: ',len(allbreak))
    else:
        #print("###########################################################")
        for i in range(0,len(allbreak)):
            
            allbreak[i].pop()
        
        allbreak=list(itertools.chain.from_iterable(allbreak))
    numericalallbreak1=[]
   
    for i in range(0,len(allbreak)):
        frag=[]
        for k in range(0,len(allbreak[i])):
            breakedge=[0]*13
            breakedge[11]=[allbreak[i][k][11]]
            breakedge[12]=[allbreak[i][k][12]]
            for j in [0,1,3,5]:
                if pd.isnull(allbreak[i][k][j]):
                    breakedge[j]=[0,300]
                else:
                    breakedge[j]=[allbreak[i][k][j],0]
            for j in [2,4,6,7,8,9,10]:
                if pd.isnull(allbreak[i][k][j]):
                    breakedge[j]=[0,0,0,0,0,0,0,0,0,300]
                if allbreak[i][k][j]=="Longitudinal":
                    breakedge[j]=[1,0,0,0,0,0,0,0,0,0]
                if allbreak[i][k][j]=="Oblique" or allbreak[i][k][j]=="oblique":
                    breakedge[j]=[0,1,0,0,0,0,0,0,0,0]
                if allbreak[i][k][j]=="obtuse":
                    breakedge[j]=[0,0,1,0,0,0,0,0,0,0]
                if allbreak[i][k][j]=="acute":
                    breakedge[j]=[0,0,0,1,0,0,0,0,0,0]
                if allbreak[i][k][j]=="Spiral":
                    breakedge[j]=[0,0,0,0,1,0,0,0,0,0]
                if allbreak[i][k][j]=="Transverse":
                    breakedge[j]=[0,0,0,0,0,1,0,0,0,0]
                if allbreak[i][k][j]=="right":
                    breakedge[j]=[0,0,0,0,0,0,1,0,0,0]
                if allbreak[i][k][j]=="absent" or allbreak[i][k][j]=="Absent" or allbreak[i][k][j]=="no" or allbreak[i][k][j]=="No":
                    breakedge[j]=[0,0,0,0,0,0,0,1,0,0]
                if allbreak[i][k][j]=="present" or allbreak[i][k][j]=="Present" or allbreak[i][k][j]=="yes":
                    breakedge[j]=[0,0,0,0,0,0,0,0,1,0]
            frag+=[breakedge]

        numericalallbreak1+=[frag]
    #for i in range (0,len(numericalallbreak1)):
    #    if 0 in numericalallbreak1[i]:
    #        print("F",i,numericalallbreak1[i])
    #    if "Block" in numericalallbreak1[i]:
    #        print(numericalallbreak1[i])
    numericalallbreak=[]
 
    for z in range(0,len(numericalallbreak1)):
        frag=[]
        for i in range(0,len(numericalallbreak1[z])):
            x=numericalallbreak1[z][i]
            x=list(itertools.chain.from_iterable(x))
            frag+=[x]
        numericalallbreak+=[frag]
    cleandata=numericalallbreak
    
    return cleandata,classtype   


def numerical_Label(cleandata,classtype):
    Cleandata=[]
    for i in range(0,len(classtype)):
        for j in range(0,len(cleandata)):
            #print(cleandata[j])
            if str(cleandata[j][-1])==classtype[i]:
                #print("true") 
                cleandata[j].pop()
                cleandata[j].pop()
                cleandata[j]+=[i]
                Cleandata+=[cleandata[j]]
            #print(cleandata[j]) 
    return Cleandata


def split(cleandata,split_percentage):
    
    ##########data split ####################################
    #sm=SMOTE(k_neighbors=3)
    xtrain=[]
    ytrain=[]
    frag_label=[]
    for i in cleandata:
        ytrain+=[i[0][-1]]
        xtrain+=[i]

    xtrain1,xtest1,ytrain1,ytest1=train_test_split(xtrain,ytrain,test_size=split_percentage)
    ytrain1=[]
    ytest1=[]
    #xtrain1,ytrain1=sm.fit_resample(xtrain1,ytrain1)

    xtrain1=list(itertools.chain.from_iterable(xtrain1))
    xtest1=list(itertools.chain.from_iterable(xtest1))
    for i in range(0,len(xtrain1)):
        ytrain1+=[xtrain1[i][-1]]
        xtrain1[i].pop()
        xtrain1[i].pop()
    
    for i in range(0,len(xtest1)):
        ytest1+=[xtest1[i][-1]]
        frag_label+=[xtest1[i][-2]]
        xtest1[i].pop()
        xtest1[i].pop()
        

    #print(ytest1)
    #print(frag_label)
    return xtrain1,xtest1,ytrain1,ytest1,frag_label

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if curr_frequency> counter: 
            counter = curr_frequency 
            num = i 
  
    return num



def train_classifier(xtrain,ytrain):
    clf=ExtraTreesClassifier(n_estimators=150)
    clf.fit(xtrain,ytrain)
    return clf

def curve_prediction(xtest,frag_label,clf):
    bone=[]
    for i in range (0,len(frag_label)):
        if frag_label[i] not in bone:
            bone+=[frag_label[i]]
            
    bone=[[i] for i in bone]
    
    for i in range(0,len(xtest)):
        for j in range(0,len(bone)):
            if frag_label[i]==bone[j][0]:
                bone[j]+=clf.predict([xtest[i]]).tolist()
    
    return bone

def voting(prediction):
    Result=[]
    for i in range(0,len(prediction)):
        Result+=[[prediction[i][0],str(most_frequent(prediction[i][1:]))]]
    return Result

def accuracy(Result,key,classtype):
    labelchecked=[]
    accurate=[0]*(2*len(classtype))
    for i in range(0,len(Result)):
        for j in range(0,len(key)):
            if Result[i][0]==key[j][0] and (Result[i][0] not in labelchecked):
                for g in range(1,len(classtype)+1):
                    if Result[i][1]==classtype[g-1]:
                        print([Result[i][1],key[j][1]])
                        if Result[i][1]==str(key[j][1]):
                            accurate[2*g-2]+=1
                            labelchecked+=[Result[i][0]]
                        else:
                            accurate[2*g-1]+=1
                            labelchecked+=[Result[i][0]]
    TRUE=0
    FALSE=0
    for i in range(1,len(classtype)+1):
        if (accurate[2*i-2]+accurate[2*i-1])!=0:
            print('Class ',classtype[i-1],' has ',accurate[2*i-2],' bones correctly predicted; ',accurate[2*i-1], ' bones wrongly predicted.')
            print('Class accuracy: ',(accurate[2*i-2])/(accurate[2*i-2]+accurate[2*i-1]))
            TRUE+=accurate[2*i-2]
            FALSE+=accurate[2*i-1]
    print('####################################################')
    print('Total accuracy: ',TRUE/(TRUE+FALSE))




if __name__ == "__main__":
    if len(sys.argv) >= 5:
        xls_name = sys.argv[1]
        variable_name = sys.argv[2]
        split_percentage = float(sys.argv[3])
        boo=sys.argv[4]
        
    else:
        sys.exit('Usage: python Data_auto_process.py <xls file> <Element> <Split percentage> <Balance> <removeleastclass>')
    if boo=='T':
        balance=True
    else:
        balance=False
  
    #allbreak contains all the categorical data of avaliable break curves
    fragments,key=Loaddata(xls_name, variable_name)
    leaveoneout(fragments,key,balance)
    '''
    #cleandata is all the converted(numerical) data of break curves
    #classtype is a list that keeps all the class names
    cleandata,classtype=balancing_converting(fragments,balance)
    
    #cleandata=numerical_Label(cleandata,classtype)
    

    #xtrain,ytrain is training data, xtest contains label of bones, be careful
    xtrain,xtest,ytrain,ytest,frag_label=split(cleandata,split_percentage)
    #print(xtrain[0],ytrain[0],xtest[0])
    
    #clf is the trained classifier
    clf=train_classifier(xtrain,ytrain)
    #prediction is a list in form of [bone label, prediction1, prediction2 ...]
    prediction=curve_prediction(xtest,frag_label,clf)

    #Result is a list in form of [bone label, prediction]
    Result=voting(prediction)

    
    #accuracy prints accuracy of each class and total accuracy
    accuracy(Result,key,classtype)
    '''
