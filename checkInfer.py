import os, json

def read(path):
    with open(path, "r") as f:
        return f.readlines()


filePath="./tmp/output.csv"
fileDataList=list(read(filePath))
unitDiv=1e-6
refValueFile="./tmp/tensorOut.csv"
refFileData=list(read(refValueFile))

headerStr=fileDataList.pop(0)
headerList=headerStr.strip().split(",")
headerList.pop(0)
print(headerList)

dataDict={}

for line in fileDataList:
    lineSplit=line.strip().split(",\"")
    lineHeader=lineSplit.pop(0)
    row=lineHeader.split("_")[1]
    col=lineHeader.split("_")[2]
    dataDictKey=row+"_"+col
    if not dataDictKey in dataDict:
        dataDict[dataDictKey]={}
    sign=lineHeader.split("_")[3]
    print(sign)
    count0=0
    dict0={}
    for part in lineSplit:
        y=part.strip("\"")
        dict0[headerList[count0]]=eval(y)
        count0+=1
    dataDict[dataDictKey][sign]=dict0

with open("./tmp/result.json", 'w') as fout:
    json_dumps_str = json.dumps(dataDict, indent=4)
    print(json_dumps_str, file=fout)

dataDict1={}

for array in dataDict:
    arrayData=dataDict[array]
    for sign in arrayData:
        if sign=="pos":
            mult=1
        elif sign=="neg":
            mult=-1
        signData=arrayData[sign]
        for Icol in signData:
            IcolData=signData[Icol]
            count1=0
            for I in IcolData:
                processName="P"+str(count1)
                if not processName in dataDict1:
                    dataDict1[processName]={}
                if not Icol in dataDict1[processName]:
                    dataDict1[processName][Icol]={
                                                    "dataList":[],
                                                    "dataSum":0
                                                    }
                calc=(mult*I)/unitDiv
                dataDict1[processName][Icol]["dataList"].append(calc)
                dataDict1[processName][Icol]["dataSum"]+=calc
                count1+=1

dataDict2={}
dataWrite=open("InferenceCheck_Output.csv", "w")
dataWrite.write("Original,Simulation,Check\n")
count2=0
passCount=0
failCount=0
for process in dataDict1:
    maxValue=0
    maxKey=""
    processData=dataDict1[process]
    for data in processData:
        if processData[data]["dataSum"]>maxValue:
            maxValue=processData[data]["dataSum"]
            maxKey=data
    dataDict2[process]={
                        "key":maxKey,
                        "value":maxValue
                        }
    refSoftInferData=refFileData[count2].strip().split(",")[0].strip()
    simulatedData=maxKey.strip().replace("IPRB","").replace(":in","")
    if refSoftInferData==simulatedData:
        check="PASS"
        passCount+=1
    else:
        check="FAIL"
        failCount+=1
    dataWrite.write(refSoftInferData+","+simulatedData+","+check+"\n")
    count2+=1
dataWrite.write("\n\nPASS "+str(passCount)+"\nFAIL "+str(failCount))
dataWrite.close()

with open("./tmp/result2.json", 'w') as fout:
    json_dumps_str = json.dumps(dataDict1, indent=4)
    print(json_dumps_str, file=fout)
            
with open("./tmp/result3.json", 'w') as fout:
    json_dumps_str = json.dumps(dataDict2, indent=4)
    print(json_dumps_str, file=fout)
