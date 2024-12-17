from pyDtOO import dtDeveloping
from pyDtOO import dtForceDeveloping
from pyDtOO import dtScalarDeveloping
from pyDtOO import dtClusteredSingletonState as stateCounter
import numpy as np
import os
import time
import logging

class tistosRunner:
  import sys
    
  OMEGA = 7.53982
  DHZUL = -2.4
  
  def __init__(self, case, prefix, stateNumber, coeff):
    self.case_ = case
    self.prefix_ = prefix
    self.stateNumber_ = stateNumber
    self.coeff_ = coeff
    self.state_ = str(self.prefix_)+'_'+str(self.stateNumber_)
    
    self.weights_ = {"tl":1/3, "n":1/3, "vl":1/3}
    
    
    #
    # Init variables
    #
    
    self.P_ = {"tl":0, "n":0, "vl":0}
    self.dH_ = {"tl":0, "n":0, "vl":0}
    self.eta_ = {"tl":0, "n":0, "vl":0}
    self.Vcav_ = {"tl":0, "n":0, "vl":0}
    
    self.addDataList_ = [self.P_, self.dH_, self.eta_, self.Vcav_] #Define for new case...
    
    self.isOk_ = False
      
  @staticmethod
  def TransformPre(stateLabel, x, cVstr):
    return x, cVstr
  
  @staticmethod
  def DoF():
    return [
      {'label': 'cV_ru_alpha_1_ex_0.0', 'min': -0.155, 'max': 0.025}, 
      {'label': 'cV_ru_alpha_1_ex_0.5', 'min': -0.19, 'max': -0.01}, 
      {'label': 'cV_ru_alpha_1_ex_1.0', 'min': -0.19, 'max': -0.01}, 
      {'label': 'cV_ru_alpha_2_ex_0.0', 'min': -0.08, 'max': 0.1}, 
      {'label': 'cV_ru_alpha_2_ex_0.5', 'min': -0.08, 'max': 0.1}, 
      {'label': 'cV_ru_alpha_2_ex_1.0', 'min': -0.08, 'max': 0.07}, 
      {'label': 'cV_ru_offsetM_ex_0.0', 'min': 1.0, 'max': 1.5},
      {'label': 'cV_ru_offsetM_ex_0.5', 'min': 1.0, 'max': 1.5}, 
      {'label': 'cV_ru_offsetM_ex_1.0', 'min': 1.0, 'max': 1.5}, 
      {'label': 'cV_ru_ratio_0.0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_ratio_0.5', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_ratio_1.0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_offsetPhiR_ex_0.0', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_offsetPhiR_ex_0.5', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_offsetPhiR_ex_1.0', 'min': -0.15, 'max': 0.15}, 
      {'label': 'cV_ru_bladeLength_0.0', 'min': 0.4, 'max': 0.8}, 
      {'label': 'cV_ru_bladeLength_0.5', 'min': 0.6, 'max': 1.0}, 
      {'label': 'cV_ru_bladeLength_1.0', 'min': 0.8, 'max': 1.3}, 
      {'label': 'cV_ru_t_le_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_le_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_le_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_mid_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_mid_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_mid_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_te_a_0', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_t_te_a_0.5', 'min': 0.005, 'max': 0.06},
      {'label': 'cV_ru_t_te_a_1', 'min': 0.005, 'max': 0.06}, 
      {'label': 'cV_ru_u_mid_a_0', 'min': 0.4, 'max': 0.6}, 
      {'label': 'cV_ru_u_mid_a_0.5', 'min': 0.4, 'max': 0.6},
      {'label': 'cV_ru_u_mid_a_1', 'min': 0.4, 'max': 0.6}
      ]
  
  def ReadResults(self):
    omega = tistosRunner.OMEGA
    s = stateCounter(self.stateNumber_)
    histDict = s.readDict('history')
    try:
          
      for i in ("tl" , "n", "vl"):
        caseDirOf = self.case_+"_"+i+"_"+self.state_
        time_step_holder = 1000
        #
        # read openFoam results
        #
        Q_ru_dev = dtScalarDeveloping( dtDeveloping(
          os.path.join(caseDirOf, 'postProcessing/Q_ru_in/100')
          ).Read() )
        print("Q_ru_dev")
        pIn_ru_dev = dtScalarDeveloping( dtDeveloping(
          os.path.join(caseDirOf, 'postProcessing/ptot_ru_in/100')
          ).Read() )
        print("pIn_ru_dev")
        pOut_ru_dev = dtScalarDeveloping( dtDeveloping(
          os.path.join(caseDirOf, 'postProcessing/ptot_ru_out/100')
          ).Read() )
        print("Vcav_ru_dev")
        Vcav = dtScalarDeveloping( dtDeveloping(
          os.path.join(caseDirOf, 'postProcessing/V_CAV/100')
          ).Read() )
        print("fdev")
        F_dev = dtForceDeveloping( dtDeveloping(
          os.path.join(caseDirOf, 'postProcessing/forces')
          ).Read({'force.dat' : ':,4:10', 'moment.dat' : ':,4:10', '*.*' : ''}) )

        average_time = int(time_step_holder/10)
        self.P_[i] = F_dev.MomentMeanLast(average_time)[2] * omega
        self.dH_[i] = (pOut_ru_dev.MeanLast(average_time) - pIn_ru_dev.MeanLast(average_time)) / 9.81
        self.eta_[i] = self.P_[i] / (1000. * 9.81 * self.dH_[i] * Q_ru_dev.MeanLast(average_time) )
        self.Vcav_[i] = Vcav.MeanLast(average_time)
        
        # check for last iteration
        self.lastIt_ = int( pIn_ru_dev.LastTime() )
        if (self.lastIt_ != time_step_holder):
          raise ValueError('Max number of iterations not reached.')
          
        # Throw an error in case of a pump
        if np.abs(self.eta_[i]) > 1:
          raise ValueError('Pump detected.')
            
         
      #
      # toggle isOk to True
      #
      
      self.isOk_ = True
      histDict['DateOfEvaluation'] = time.ctime()
      s.update('history', histDict)
              
    except Exception as e:
      logging.warning('Catch exception : %s', e)
      histDict['DateOfFailure'] = time.ctime()
      histDict['ReasonOfFailure'] = str(e)
      s.update('history', histDict)
              
  @staticmethod
  def FailedFitness():
    return [tistosRunner.sys.float_info.max]*3
  
  @staticmethod
  def IsFailedFitness(fit):
    if fit[0] == tistosRunner.sys.float_info.max:
      return True
    else:
      return False
      
  def GiveFitness(self):
    
    dHZul = tistosRunner.DHZUL
    fitnessEta, fitnessVcav, fitnessH = 0.0, 0.0, 0.0
    
    if self.isOk_:
      
      #
      # update databse except islandID
      #
    
      for addDataIndex in range(4):
        name = stateCounter.ADDDATA[addDataIndex]
        stateCounter(self.stateNumber_).update(name, self.addDataList_[addDataIndex])
      
      #
      # calculate fitness
      #
      
      # deviation from design head
      devH = np.abs(self.dH_['n']/dHZul -1)
      
      # deviation from design head must be <= 20%, otherwise individual
      # is invalid. Multiplication of penalty term with deviation from design
      # head --> If two indivuduals are invalid, the individual with the lower
      # violation of the constraint is dominant.
      if devH > 0.2:
        return [1e6*devH]*3
      

      for i in ("tl", "n", "vl"):
        fitnessEta += self.weights_[i] * np.abs(1+self.eta_[i]) 
        fitnessVcav += self.weights_[i] * self.Vcav_[i] 
      
      fitnessH = np.abs(self.dH_['n'] - dHZul)
      
      return [fitnessEta, fitnessVcav, fitnessH]
    
    else:
      return tistosRunner.FailedFitness()
          
    



                    
                    

                
                
                
            
            
