# This file is the data processing for CDPO.
# Author: Yiju Guo
# Date: 2024-01
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

import json

# Convert objects to json strings
def changeToJson(obj,isFormat)->str:
    if(not isFormat):
        return json.dumps(obj,ensure_ascii=False, separators=(',', ':'))    
    else:
        return json.dumps(obj,indent=4, sort_keys=True, separators=(',', ':'),ensure_ascii =False)


def writefile(filePath,fileStr:str,encoding= "utf-8",callBack = None):
    with open(filePath, "w", encoding = encoding) as file:
        if callBack != None:
            callBack(file)
        fileStr.encode(encoding)
        file.write(fileStr)
        file.close()

def writeJsonFile(filePath:str,jsonObj,encoding = "utf-8",isFormat = True,isForceJson = False):
    if (not jsonObj or not filePath.endswith(".json")) and not isForceJson:
        print("writeJsonFile Error: Invalid arguments",flush=True)
        exit(-1)
    writefile(filePath,changeToJson(jsonObj,isFormat),encoding)


def readJsonString(jsonStr:str)->dict:
    return json.loads(jsonStr)
    

def readfile(filePath,encoding= "utf-8",callBack = None):
    tempStr = ''
    import os
    current_path = os.getcwd()
    print("Current path：", current_path)
    with open(filePath, "r", encoding = encoding) as file:
        lines = file.readlines()
        if callBack != None:
            callBack(lines,file)
        tempStr = tempStr.join(lines)
        file.close()
    return tempStr


def readJsonFile(filePath:str,encoding = "utf-8",isForceJson = False)->dict:
    if not filePath.endswith(".json") and not isForceJson:
        print("readJsonFile Error: filePath is not Json File")
        exit(-1)
    jsonObj = None
    jsonObj = readJsonString(readfile(filePath,encoding)) 
    if not jsonObj:
        print("readJsonFile Error: jsonObj is not NULL")
        exit(-1)
    return jsonObj