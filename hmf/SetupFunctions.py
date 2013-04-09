'''
Created on Apr 20, 2012

@author: Steven Murray
@contact: steven.murray@uwa.edu.au

Last Updated: 12 March 2013

This module contains 4 functions which all have the purpose of importing the
transfer function (or running CAMB on the fly) for
input to the Perturbations class (in Perturbations.py)

The functions are:
    1. SetParameter: modifies the params.ini file for use in CAMB with specified parameters
    2. ImportTransferFunction: Merely reads a CAMB transfer file and outputs ln(k) and ln(T).
    3. CAMB: runs CAMB through the OS (camb must be compiled previously)
    4. Setup: A driver for the above functions. Returns ln(k) and ln(T)
'''

###############################################################################
# Some simple imports
###############################################################################
import numpy as np
import os
import time

import pycamb

###############################################################################
# The function definitions
###############################################################################        
def SetParameter(filename,keys):
    """
    Sets parameters defined in the dictionary 'keys' into the CAMB params file 'filename'
    
    INPUT:
    filename: name of the ini file for CAMB. Assumes that the current directory is the camb folder.
    keys: a dictionary of CAMB parameters, with the keys being the variable names in CAMB and the values their values
    
    OUTPUT:
    None
    """
    #Open the file 
    file_object = open(filename,'r+')
        
    #file_data is a list of the lines in the parameter file
    file_data = file_object.readlines()
    for parameter,value in keys.iteritems():
        for number,line in enumerate(file_data):
            #Find a line that starts with the name of the parameter to be set
            if line.strip().startswith(parameter):
                if type(value) is type("string"):
                    file_data[number] = line.replace(line.partition('=')[2].strip(),value) 
                elif type(value) is type(True):
                    if value:
                        file_data[number] = line.replace(line.partition('=')[2].strip(),'T')
                    else:
                        file_data[number] = line.replace(line.partition('=')[2].strip(),'F')
                else:
                    file_data[number] = line.replace(line.partition('=')[2].strip(),str(value))
                    
                break
                
                
    file_object.seek(0)
    file_object.writelines(file_data)
    file_object.close()
    
def ImportTransferFunction(transfer_file):
    """
    Imports the Transfer Function file to be analysed, and returns the pair ln(k), ln(T)
    
    Input: "transfer_file": full path to the file containing the transfer function (from camb).
    
    Output: ln(k), ln(T)
    """
     
    transfer = np.loadtxt(transfer_file)
    k = transfer[:,0]
    T = transfer[:,1]   
    #k,T = TableIO.readColumns(transfer_file,"!#",columns=[0,1])
  
    k = np.log(k)
    T = np.log(T)


    return k,T
        


def CAMB(camb_dict,prefix):
    """
    Uses CAMB in its current setup to produce a transfer function
    
    The function needs to be imported by calling ImportTransferFunction.
    """
    #Get current time as a format string
    if prefix is None:
        prefix = time.asctime( time.localtime(time.time())).replace(" ","").replace(":","")
    
    camb_dict["output_root"] = prefix
        
    os.chdir('camb')
    os.system("cp HMF_params.ini params_"+prefix+'.ini')
    SetParameter('params_'+prefix+'.ini', camb_dict)
    os.system('module load gfortran')
    os.system('./camb params_'+prefix+'.ini')
    os.system('rm params_'+prefix+'.ini')
    os.chdir('..')
    
    return camb_dict
    
def Setup(transfer_file,camb_dict,prefix=None):
    """
    A convenience function used to fully setup the workspace in the 'usual' way
    """
    #If no transfer file uploaded, but it was custom, execute CAMB
    if transfer_file is None:
        #camb_dict = CAMB(camb_dict,prefix)
        k,T,sig8 = pycamb.transfers(**camb_dict)
        T = np.log(T[1,:,0])
        k = np.log(k)
        del sig8
        #transfer_file = 'camb/'+camb_dict["output_root"]+'_transfer_out.dat'

    else:
        #Import the transfer file wherever it is.
        k,T = ImportTransferFunction(transfer_file)
    
#    #If we just created it with CAMB, now delete it and other bits that have been made.
#    if transfer_file.endswith('_transfer_out.dat'):
#        os.system('rm -f '+camb_dict["output_root"]+'*')
#    
    return k,T
    
