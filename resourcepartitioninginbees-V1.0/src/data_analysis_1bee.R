library(plyr) # count
library(rstudioapi) # getActiveDocumentContext
library(RColorBrewer) # colorRampPalette
library(bipartite) # H2fun, DIRT_LPA_wb_plus
library(beepr) # beep
library(MASS) # ?
library(Rcpp) # sourceCpp
library(lme4) # glmer
library(emmeans) # lstrends
library(ggplot2)
library(ggthemes)
library(gtable)
library(grid)
library(gridExtra)
library(tibble)
library(car)
library(ggpubr)

rm(list=ls());

currentPath = getActiveDocumentContext()$path
setwd(dirname(currentPath))

# Functions & Sourcing ----------------------------------------------------



SinkGLMEROutput = function(name,testOutput,path,hoc=NULL)
{
  file.create(paste(path,"/",name,".txt",sep=""))
  sink(paste(path,"/",name,".txt",sep=""))
  print(summary(testOutput))
  cat("\n\n")
  print(anova(testOutput))
  cat("\n\n")
  if(!is.null(hoc)) print(hoc);
  sink(NULL)
}

source("01-Functions.R")


# Parameters --------------------------------------------------------------



# Specify the folder & path associated. getwd() returns already the path until the root. Only specify the path from the root.
simulationToAnalyse = "/Output";

# Simulation specifications
numberOfArrays = 1;
numberOfSimulations = 500;
numberOfBouts = 40;
numberOfBees = 1;
# numberOfResources = 10;

arrayOfTest = c("plos")

overwriteFiles = FALSE;

makeFilm = FALSE; # Create a 2D density plot for each array and all arrays combined

# SubSeqSimilarity specific parameters
iter = 100;
subSeqSize = 3;
stopAfterTrapline = F;

# Root Code (Run before any part) -----------------------------------------

# Get to the specified path and retrieve the file in this folder.
outputDirectory = paste(getwd(),simulationToAnalyse,sep="");
testFolders = list.files(path=outputDirectory);
# By default we retrieve files that contain the "generate" word. 
testFolders = testFolders[CharacterMatch(testFolders,"plos")];


if (length(testFolders)==0) {testFolders=c("")}

numberOfTests = length(testFolders);

# Initialize output dataframe
arrayTypesOnData = c();
learningFactors = c();
abandonFactors = c();
routeCompares = c();
useQLearning = c();
alphaPos = c();
alphaNeg = c();
betaQL = c();
gammaQL = c();
for(fld in testFolders)
{
  if (fld=="") {
    arrayInfos = read.csv(paste(outputDirectory,"/",fld,"Array01/arrayInfos.csv",sep=""))
    beeInfos = read.csv(paste(outputDirectory,"/",fld,"Array01/beeInfos.csv",sep=""))
    learningFactors = c(learningFactors,beeInfos$learningFactor[1]);
    abandonFactors = c(abandonFactors,beeInfos$abandonFactor[1]);
    useQLearning = c(useQLearning,beeInfos$useQLearning[1]);
    alphaPos = c(alphaPos,beeInfos$alphaPos[1]);
    alphaNeg = c(alphaNeg,beeInfos$alphaNeg[1]);
    betaQL =  c(betaQL,beeInfos$beta[1]);
    gammaQL =  c(gammaQL,beeInfos$gamma[1]);
    
    if(beeInfos$routeCompare[1]) routeCompares = c(routeCompares,"routeCompare") else routeCompares = c(routeCompares,"noRouteCompare");
    
    arrayNameChr = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
    
    if(!is.na(arrayInfos$flowerPerPatch)) {arrayNameChr = paste(arrayNameChr,"-",arrayInfos$flowerPerPatch,sep="")}
    
    arrayTypesOnData = c(arrayTypesOnData,arrayNameChr)
    
  }else{  
    arrayInfos = read.csv(paste(outputDirectory,"/",fld,"/Array01/arrayInfos.csv",sep=""))
    beeInfos = read.csv(paste(outputDirectory,"/",fld,"/Array01/beeInfos.csv",sep=""))
    learningFactors = c(learningFactors,beeInfos$learningFactor[1]);
    abandonFactors = c(abandonFactors,beeInfos$abandonFactor[1]);
    useQLearning = c(useQLearning,beeInfos$useQLearning[1]);
    alphaPos = c(alphaPos,beeInfos$alphaPos[1]);
    alphaNeg = c(alphaNeg,beeInfos$alphaNeg[1]);
    betaQL =  c(betaQL,beeInfos$beta[1]);
    gammaQL =  c(gammaQL,beeInfos$gamma[1]);
    
    if(beeInfos$routeCompare[1]) routeCompares = c(routeCompares,"routeCompare") else routeCompares = c(routeCompares,"noRouteCompare");
    
    arrayNameChr = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
    
    if(!is.na(arrayInfos$flowerPerPatch)) {arrayNameChr = paste(arrayNameChr,"-",arrayInfos$flowerPerPatch,sep="")}
    
    arrayTypesOnData = c(arrayTypesOnData,arrayNameChr)}
  
}

numberOfArrayTypes = length(arrayTypesOnData)

outputData = data.frame(arrayType = rep(arrayTypesOnData,each=(numberOfTests/numberOfArrayTypes)*numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        algorithm = rep(routeCompares,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        QLearning=rep(useQLearning,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        alphaPos=rep(alphaPos,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        alphaNeg=rep(alphaNeg,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        betaQL=rep(betaQL,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        gammaQL=rep(gammaQL,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        learningValue = rep(learningFactors,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        abandonValue = rep(abandonFactors,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        arrayNumber = rep(c(1:numberOfArrays),each=numberOfSimulations*numberOfBouts*numberOfBees,times=numberOfTests),
                        simulation = rep(c(1:numberOfSimulations),each=numberOfBouts*numberOfBees,times=numberOfTests*numberOfArrays),
                        bout = rep(c(1:numberOfBouts),each=numberOfBees,times=numberOfTests*numberOfArrays*numberOfSimulations),
                        bee = rep(c(1:numberOfBees),times=numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts),
                        stringsAsFactors = F);

for(testNumber in 1:length(testFolders))
{
  testFolderName = testFolders[testNumber];
  testFolderPath = paste(outputDirectory,"/",testFolderName,sep="");
}






############### similarity data ###################
#/!\ careful: can be quite heavy to run

#function to compute subsequence similarity
SubSequenceSimilarity=function(seq1,seq2,subSeqSize)
{
  l1=length(seq1);
  l2=length(seq2);
  
  if (l1<subSeqSize | l2<subSeqSize)
  {return(0)} else {
    
    listDistinctSubseq=list();
    distinctSubseqInCommon=c()
    
    nSubseq1=l1-subSeqSize+1;
    nSubseq2=l2-subSeqSize+1;
    
    seq=c(list(seq1),list(seq2));
    nSubseq=c(nSubseq1,nSubseq2);
    
    
    for (i in (1:2))
    {
      for (k in (1:nSubseq[i]))
      {
        subSeq=list(seq[[i]][k:(k+subSeqSize-1)]);
        
        matchedSubseq=match(listDistinctSubseq,subSeq);
        if (sum(matchedSubseq,na.rm=TRUE)==0)
        {
          listDistinctSubseq=c(listDistinctSubseq,subSeq);
          distinctSubseqInCommon=c(distinctSubseqInCommon,FALSE);
          
          
        }else{
          indexMatchedSubseq=which(!is.na(matchedSubseq));
          distinctSubseqInCommon[indexMatchedSubseq]=TRUE}
      }
    };
    
    visitationsInCommon=list(rep(FALSE,l1),rep(FALSE,l2));
    
    for (i in (1:2))
    {
      for (k in (1:nSubseq[i]))
      {
        subSeq=list(seq[[i]][k:(k+subSeqSize-1)]);
        indexMatchedSubseq=which(!is.na(match(listDistinctSubseq,subSeq)));
        if (distinctSubseqInCommon[indexMatchedSubseq])
        {
          visitationsInCommon[[i]][k:(k+subSeqSize-1)]=TRUE
        }
      }
    };
    
    
    
    sab=sum(visitationsInCommon[[1]])+sum(visitationsInCommon[[2]]);
    return(sab/(2*max(l1,l2)))
  }
}

#now computing the dataframe



# Assuming at all time a 95% confidence interval
lowerThreshold = iter*0.025;
upperThreshold = iter*0.975;

for(testNumber in 1:length(testFolders))
{
  testFolderName = testFolders[testNumber]; # fileName
  testFolderPath = paste(outputDirectory,"/",testFolderName,sep=""); # fileDirectory
  
  # Check if the file already exists
  testFiles = list.files(path = testFolderPath);
  if(any(testFiles=="similarityData.csv") & !overwriteFiles)
  {
    cat("Similarity computation for test ",testFolderName," is already done. Proceeding to next test.\n",sep="");
    next;
  }
  cat("Starting similarity assessment for test : ",testFolderName,".\n",sep="");
  
  arrays = list.files(testFolderPath);
  arrays = arrays[CharacterMatch(arrays,"Array")];
  
  ## Initialize output data
  similarityDF = data.frame(arrayNumber = rep(c(1:numberOfArrays),each = numberOfSimulations*(numberOfBouts-1)*numberOfBees),
                            simulation = rep(1:numberOfSimulations, each = (numberOfBouts-1)*numberOfBees, times = numberOfArrays),
                            bout = rep(1:(numberOfBouts-1),each=numberOfBees,times = numberOfArrays*numberOfSimulations),
                            bee = rep(1:numberOfBees,times = numberOfArrays*numberOfSimulations*(numberOfBouts-1)),
                            similarityIndex = 0,
                            stringsAsFactors = F);
  similarityIndexVector = numeric(numberOfArrays*numberOfSimulations*(numberOfBouts-1)*numberOfBees);
  
  lineToFill = 0;
  for(arrayNumber in 1:length(arrays))
  {
    array = arrays[arrayNumber];
    arrayDirectory = paste(testFolderPath,array,sep="/");
    
    visitationSequences = read.csv(paste(arrayDirectory,"/matrixOfVisitationSequences.csv",sep=""));
    
    for(sim in 1:numberOfSimulations)
    {
      simVisitationSequences = subset(visitationSequences,visitationSequences[,1]==sim);
      for(bout in 1:(numberOfBouts-1))
      {
        boutVisitationSequences = simVisitationSequences[which(simVisitationSequences[,2]==bout | simVisitationSequences[,2]==bout+1),];
        
        for(bee in 1:numberOfBees)
        {
          indVisitationSequences = subset(boutVisitationSequences,boutVisitationSequences[,3]==bee)[,-c(1:3)];
          
          lineToFill = lineToFill + 1;
          
          seq1 = indVisitationSequences[1,-1];
          seq1 = as.numeric(seq1)[which(seq1!=0 & seq1!=1)];
          
          seq2 = indVisitationSequences[2,-1];
          seq2 = as.numeric(seq2)[which(seq2!=0 & seq2!=1)];
          
          similarityIndexVector[lineToFill] = SubSequenceSimilarity(seq1,seq2,subSeqSize)
        }
      }
    }
  }
  
  ## Compile all data in the data.frame
  similarityDF$similarityIndex = similarityIndexVector;
  
  
  write.csv(similarityDF,paste(testFolderPath,"/similarityData.csv",sep=""),row.names = F);
}

# Join the files
similarityVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees);
lineSimilarity = 1;
outputDataSim = data.frame();

for(testNumber in 1:length(testFolders))
{
  folderName = testFolders[testNumber];
  
  fileDirectory = paste(outputDirectory,"/",folderName,sep="");
  
  ## Import SubSeqSim
  subSeqSim = read.csv(paste(fileDirectory,"/similarityData.csv",sep=""));
  
  beeInfos = read.csv(paste(fileDirectory,"/Array01/beeInfos.csv",sep=""))
  arrayInfos = read.csv(paste(fileDirectory,"/Array01/arrayInfos.csv",sep=""))
  arrayType = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
  
  extractSimilarity = subSeqSim$similarityIndex
  dataLength = length(extractSimilarity);
  similarityVector[c(lineSimilarity:(lineSimilarity+dataLength-1))] = extractSimilarity;
  lineSimilarity = lineSimilarity + dataLength;
  
  if(beeInfos$routeCompare[1]) algorithmChr = "routeCompare" else algorithmChr = "noRouteCompare"
  
  reorgData = data.frame(arrayType = arrayType,
                         algorithm = algorithmChr,
                         learningValue = beeInfos$learningFactor[1],
                         abandonValue = beeInfos$abandonFactor[1],
                         QLearning = beeInfos$useQLearning[1],
                         alphaPos = beeInfos$alphaPos[1],
                         alphaNeg = beeInfos$alphaNeg[1],
                         betaQL = beeInfos$beta[1],
                         gammaQL = beeInfos$gamma[1],
                         subSeqSim)
  
  
  outputDataSim = rbind(outputDataSim,reorgData);
}

write.csv(outputDataSim,paste(outputDirectory,"/similarityData.csv",sep=""),row.names = F)





##################### route quality data #############################


# Join the files

rawQualityVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts);
relativeQualityVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts);
lineRouteQuality = 1;

for(testNumber in 1:length(testFolders))
{
  folderName = testFolders[testNumber];
  
  fileDirectory = paste(outputDirectory,"/",folderName,sep="");
  
  arrays = list.files(fileDirectory);
  arrays = arrays[CharacterMatch(arrays,"Array")];
  
  for (array in arrays) {
    
    ## Import route Quality
    routeQuality = read.csv(paste(fileDirectory,'/',array,"/routeQualityDF.csv",sep=""));
    extractRawQuality = routeQuality$rawQuality;
    extractRelativeQuality = routeQuality$relativeQuality;
    dataLength = length(extractRawQuality);
    rawQualityVector[c(lineRouteQuality:(lineRouteQuality+dataLength-1))] = extractRawQuality;
    relativeQualityVector[c(lineRouteQuality:(lineRouteQuality+dataLength-1))] = extractRelativeQuality;
    lineRouteQuality = lineRouteQuality + dataLength;
    
  }
  
}


### Output routeQuality
routeQualityData = cbind(outputData[which(outputData$bee==1),],
                         rawQuality = rawQualityVector,
                         relativeQuality = relativeQualityVector);
write.csv(routeQualityData,paste(outputDirectory,"/routeQualityData.csv",sep=""),row.names = F)


############plots###################


#routeQuality
data=read.csv(paste(outputDirectory,"/routeQualityData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[!data$QLearning & data$learningValue==1.5 & data$abandonValue==1.]="Dubois 1"
modelVector[!data$QLearning & data$learningValue==1. & data$abandonValue==0.75]="Dubois 2"
modelVector[!data$QLearning & data$learningValue==1.5 & data$abandonValue==0.75]="Dubois 3"
modelVector[data$QLearning & data$alphaPos>0 & data$alphaNeg==0]="QLearning 1"
modelVector[data$QLearning & data$alphaPos==0 & data$alphaNeg>0]="QLearning 2"
modelVector[data$QLearning & data$alphaPos>0 & data$alphaNeg>0]="QLearning 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data_mean=aggregate(data,list(data$model,data$bout),mean)
data_sd= aggregate(data,list(data$model,data$bout),sd)
data=data_mean
data["sd"]=data_sd["relativeQuality"]
ggplot(data = data,aes(x=bout, y=relativeQuality)) +
  geom_line()+
  facet_wrap(~Group.1)+
  geom_errorbar(aes(ymin=relativeQuality-sd, ymax=relativeQuality+sd), width=.2, position=position_dodge(0.05))



#similarity index
data=read.csv(paste(outputDirectory,"/similarityData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[!data$QLearning & data$learningValue==1.5 & data$abandonValue==1.]="Dubois 1"
modelVector[!data$QLearning & data$learningValue==1. & data$abandonValue==0.75]="Dubois 2"
modelVector[!data$QLearning & data$learningValue==1.5 & data$abandonValue==0.75]="Dubois 3"
modelVector[data$QLearning & data$alphaPos>0 & data$alphaNeg==0]="QLearning 1"
modelVector[data$QLearning & data$alphaPos==0 & data$alphaNeg>0]="QLearning 2"
modelVector[data$QLearning & data$alphaPos>0 & data$alphaNeg>0]="QLearning 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data_mean=aggregate(data,list(data$model,data$bout),mean)
data_sd= aggregate(data,list(data$model,data$bout),sd)
data=data_mean
data["sd"]=data_sd["similarityIndex"]
ggplot(data = data,aes(x=bout, y=similarityIndex)) +
  geom_line()+
  facet_wrap(~Group.1)+
  geom_errorbar(aes(ymin=similarityIndex-sd, ymax=similarityIndex+sd), width=.2, position=position_dodge(0.05))

