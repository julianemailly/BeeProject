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
numberOfArrays = 10;
numberOfSimulations = 100;
numberOfBouts = 50;
numberOfBees = 2;
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
testFolders = testFolders[CharacterMatch(testFolders,"generate")];

# testMatch = rep(F,length(testFolders))
# for(val in 1:length(arrayOfTest))
# {
#   testMatch[which(CharacterMatch(testFolders,arrayOfTest[val]))] = T
# }
# testFolders = testFolders[testMatch]

if (length(testFolders)==0) {testFolders=c("")}

numberOfTests = length(testFolders);

# Initialize output dataframe
arrayTypesOnData = c();
learningFactors = c();
abandonFactors = c();
routeCompares = c();
for(fld in testFolders)
{
  if (fld=="") {
    arrayInfos = read.csv(paste(outputDirectory,"/",fld,"Array01/arrayInfos.csv",sep=""))
    beeInfos = read.csv(paste(outputDirectory,"/",fld,"Array01/beeInfos.csv",sep=""))
    learningFactors = c(learningFactors,beeInfos$learningFactor[1]);
    abandonFactors = c(abandonFactors,beeInfos$abandonFactor[1]);
    if(beeInfos$routeCompare[1]) routeCompares = c(routeCompares,"routeCompare") else routeCompares = c(routeCompares,"noRouteCompare");
    
    arrayNameChr = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
    
    if(!is.na(arrayInfos$flowerPerPatch)) {arrayNameChr = paste(arrayNameChr,"-",arrayInfos$flowerPerPatch,sep="")}
    
    arrayTypesOnData = c(arrayTypesOnData,arrayNameChr)
  }else{  
  arrayInfos = read.csv(paste(outputDirectory,"/",fld,"/Array01/arrayInfos.csv",sep=""))
  beeInfos = read.csv(paste(outputDirectory,"/",fld,"/Array01/beeInfos.csv",sep=""))
  learningFactors = c(learningFactors,beeInfos$learningFactor[1]);
  abandonFactors = c(abandonFactors,beeInfos$abandonFactor[1]);
  if(beeInfos$routeCompare[1]) routeCompares = c(routeCompares,"routeCompare") else routeCompares = c(routeCompares,"noRouteCompare");
  
  arrayNameChr = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
  
  if(!is.na(arrayInfos$flowerPerPatch)) {arrayNameChr = paste(arrayNameChr,"-",arrayInfos$flowerPerPatch,sep="")}
  
  arrayTypesOnData = c(arrayTypesOnData,arrayNameChr)}
  
}

numberOfArrayTypes = length(arrayTypesOnData)

outputData = data.frame(arrayType = rep(arrayTypesOnData,each=(numberOfTests/numberOfArrayTypes)*numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
                        algorithm = rep(routeCompares,each=numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees),
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









########### competition data #####################

for(testNumber in 1:length(testFolders))
{
  # Set the folder path and retrieve all the subfolders necessary.
  testFolderName = testFolders[testNumber];
  testFolderPath = paste(outputDirectory,"/",testFolderName,sep="");
  if (testFolderName==""){testFolderPath=outputDirectory}
  testFiles = list.files(testFolderPath);
  
  # Check if Competition is already computed for this test
  if(any(testFiles=="competitionData.csv") & !overwriteFiles) 
  {
    cat("Competition computation for test ",testFolderName," is already done. Proceeding to next test.\n",sep="");
    next;
  }
  cat("Starting competition assessment for test : ",testFolderName,".\n",sep="");
  
  testFiles = testFiles[CharacterMatch(testFiles,"Array")];
  
  ## Retrieve the informations about the simulation
  beeInfos = read.csv(paste(testFolderPath,"/Array01/beeInfos.csv",sep=""));
  arrayType = beeInfos$arrayID[1];
  
  ## Create a dataframe that contains all quality values of all arrays
  lengthOfDF = numberOfArrays*numberOfSimulations*numberOfBouts;
  allCompetitionDF = data.frame(arrayNumber = rep(1:numberOfArrays,each = numberOfSimulations*numberOfBouts),
                                simulation = rep(1:numberOfSimulations,times=numberOfArrays,each=numberOfBouts),
                                bout = rep(1:numberOfBouts,times = numberOfArrays*numberOfSimulations),
                                interference = numeric(lengthOfDF),
                                exploitation1 = numeric(lengthOfDF),
                                exploitation2 = numeric(lengthOfDF));
  
  allInterf = numeric(lengthOfDF);
  allExploit1 = numeric(lengthOfDF);
  allExploit2 = numeric(lengthOfDF);
  
  k = 0;
  for(test in testFiles)
  {
    arrayDirectory = paste(testFolderPath,test,sep="/");
    
    ## Retrieve the visitation sequence file
    visitSeqData = read.csv(paste(arrayDirectory,"/matrixOfVisitationSequences.csv",sep=""));
    
    ## Create the new dataframe to output the differential quality
    competitionDF = data.frame(simulation = rep(1:numberOfSimulations,each=numberOfBouts),
                               bout = rep(1:numberOfBouts,times=numberOfSimulations),
                               interference = numeric(numberOfSimulations*numberOfBouts),
                               exploitation1 = numeric(numberOfSimulations*numberOfBouts),
                               exploitation2 = numeric(numberOfSimulations*numberOfBouts))
    
    arrayInterf = numeric(numberOfSimulations*numberOfBouts);
    arrayExploit1 = numeric(numberOfSimulations*numberOfBouts);
    arrayExploit2 = numeric(numberOfSimulations*numberOfBouts);
    
    ## Compute the proportion of competition occurences, and output it in both competitionDF and allCompetitionDF
    i = 0;
    for(sim in 1:numberOfSimulations)
    {
      ## Subset the competitionDF for each sim
      simCompetitionDF = subset(visitSeqData,visitSeqData[,1]==sim);
      
      for(bout in 1:numberOfBouts)
      {
        i = i + 1;
        k = k + 1;
        
        boutCompetitionDF = subset(simCompetitionDF,simCompetitionDF[,2]==bout)[,-c(1:4)];
        
        ## Compute the differential quality and output it into the new dataframe
        
        isFlowerEmpty = logical(10);
        
        interferenceOccurences = 0;
        exploitation1Occurences = 0;
        exploitation2Occurences = 0;
        
        for(visit in 1:length(boutCompetitionDF[1,]))
        {
          # Retrieve the position of bees
          visit1 = boutCompetitionDF[1,visit];
          visit2 = boutCompetitionDF[2,visit];
          
          # If both have finished their bout, finish the scan.
          if(all(boutCompetitionDF[,visit]==0)) break;
          
          # If they are on the same flower, +1 Interference
          if(visit1==visit2 && visit1!=1) 
          {
            interferenceOccurences = interferenceOccurences + 1;
            next;
          }
          
          # Check Exploitation Occurence for Ind 1
          if(visit1!=1 & visit1!=0)
          {
            if(isFlowerEmpty[visit1-1])
            {
              exploitation1Occurences = exploitation1Occurences + 1;
            } else {
              isFlowerEmpty[visit1-1] = TRUE;
            }
          }
          
          # Check Exploitation Occurence for Ind 2
          if(visit2!=1 & visit2!=0)
          {
            if(isFlowerEmpty[visit2-1])
            {
              exploitation2Occurences = exploitation2Occurences + 1;
            } else {
              isFlowerEmpty[visit2-1] = TRUE;
            }
          }
        }
        
        arrayInterf[i] = interferenceOccurences;
        arrayExploit1[i] = exploitation1Occurences;
        arrayExploit2[i] = exploitation2Occurences;
        
        allInterf[k] = interferenceOccurences;
        allExploit1[k] = exploitation1Occurences;
        allExploit2[k] = exploitation2Occurences;
      }
    }
    
    competitionDF$interference = arrayInterf;
    competitionDF$exploitation1 = arrayExploit1;
    competitionDF$exploitation2 = arrayExploit2;
    
    aggMean = aggregate(competitionDF[,3:5],by=list(bout=competitionDF$bout),FUN=mean);
    
    ## Save the dataframe (into a .csv file and into a bigger dataframe that will contain all arrays)
    write.csv2(competitionDF,paste(arrayDirectory,"/competitionData.csv",sep=""),row.names = F);
  }
  
  allCompetitionDF$interference = allInterf;
  allCompetitionDF$exploitation1 = allExploit1;
  allCompetitionDF$exploitation2 = allExploit2;
  
  ## Save the dataframe
  write.csv2(allCompetitionDF,paste(testFolderPath,"/competitionData.csv",sep=""),row.names = F);
}

# Join the files
compInterfVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees)
compExploitVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts*numberOfBees)
lineComp = 1;

for(testNumber in 1:length(testFolders))
{
  folderName = testFolders[testNumber];
  fileDirectory = paste(outputDirectory,"/",folderName,sep="");

  ## Import Competition
  competition = read.csv2(paste(fileDirectory,"/competitionData.csv",sep=""));

  extractCompInterf = rep(competition$interference,each=numberOfBees);
  dataLength = length(extractCompInterf);
  compInterfVector[c(lineComp:(lineComp+dataLength-1))] = extractCompInterf;

  extractCompExploit = as.numeric(unlist(t(competition[,c("exploitation1","exploitation2")])));
  dataLength = length(extractCompExploit);
  compExploitVector[c(lineComp:(lineComp+dataLength-1))] = extractCompExploit;
  lineComp = lineComp + dataLength;
}

competitionData = cbind(outputData,compInterference=compInterfVector,compExploitation=compExploitVector);
write.csv(competitionData,paste(outputDirectory,"/competitionData.csv",sep=""),row.names = F)







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
                         subSeqSim)
  
  
  outputDataSim = rbind(outputDataSim,reorgData);
}

write.csv(outputDataSim,paste(outputDirectory,"/similarityData.csv",sep=""),row.names = F)



  


#####################group route quality data #############################

for(testNumber in 1:length(testFolders))
{
  testFolderName = testFolders[testNumber]; # fileName
  testFolderPath = paste(outputDirectory,"/",testFolderName,sep=""); # fileDirectory
  
  # Check if the file already exists
  testFiles = list.files(path = testFolderPath);
  if(any(testFiles=="groupQualityData.csv") & !overwriteFiles) 
  {
    cat("Group quality computation for test ",testFolderName," is already done. Proceeding to next test.\n",sep="");
    next;
  }
  cat("Starting group quality assessment for test : ",testFolderName,".\n",sep="");
  
  beeInfos = read.csv(paste(testFolderPath,"/Array01/beeInfos.csv",sep=""));
  arrayInfos = read.csv(paste(testFolderPath,"/Array01/arrayInfos.csv",sep=""))
  arrayType = paste("R",arrayInfos$numberOfResources,"-P",arrayInfos$numberOfPatches,sep="")
  arrayType = beeInfos$arrayID[1]
  n=nchar(arrayType)
  arrayType=substring(arrayType,1,n-3)
  
  arrayFiles = list.files(testFolderPath);
  arrayFiles = arrayFiles[CharacterMatch(arrayFiles,"Array")];
  
  allQualityData = data.frame();
  
  for(arrayNumber in 1:length(arrayFiles))
  {
    arrayName = paste(arrayType,sprintf("%02d",arrayNumber),sep="_");
    
    arrayFolder = paste(getwd(),"/Arrays/",arrayName,sep="");
    
    optimalRoute2Ind = read.csv(paste(arrayFolder,"optimalRoute2Ind.csv",sep="/"));
    arrayGeometry = read.csv2(paste(arrayFolder,"/arrayGeometry.csv",sep=""));
    
    optimalQuality2Ind = as.double(optimalRoute2Ind);
    
    qualityFolder = paste(testFolderPath,"/Array",sprintf("%02d",arrayNumber),sep="");
    
    if(any(list.files(qualityFolder)=="routeQualityDF.csv"))
    {
      routeQualityDF = read.csv(paste(qualityFolder,"/routeQualityDF.csv",sep=""));
      # output : ArrayNumber, Simulation, Bout, Q1, Q2, Qall, QallRelative, Rank
      
      routeQualityData = data.frame()
      
      for(sim in 1:max(routeQualityDF$Simulation))
      {
        for(bout in 1:max(routeQualityDF$Bout))
        {
          indData = subset(routeQualityDF,Simulation==sim & Bout==bout)
          
          newLine = data.frame(ArrayNumber=arrayNumber,
                              Simulation=sim,
                               Bout=bout,
                               Q1=indData$rawQuality[1],
                               Q2=indData$rawQuality[2])
          routeQualityData = rbind(routeQualityData,newLine)
        }
      }
      
      groupQuality = routeQualityData$Q1+routeQualityData$Q2
      
      routeQualityData$Qall = round(groupQuality,3);
      routeQualityData$QallRelative = routeQualityData$Qall/optimalQuality2Ind;
      
      write.csv(routeQualityData,paste(qualityFolder,"/routeQualityData.csv",sep=""),row.names=F);
      file.remove(paste(qualityFolder,"/routeQualityDF.csv",sep=""))
    }
    routeQualityData = read.csv(paste(qualityFolder,"/routeQualityData.csv",sep=""))
    
    allQualityData = rbind(allQualityData,routeQualityData);
  }
  write.csv(allQualityData,paste(testFolderPath,"/groupQualityData.csv",sep=""),row.names = F);
}

# Join the files
groupQualityVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts);
Q1Vector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts);
Q2Vector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts);
lineGroupQuality = 1;

for(testNumber in 1:length(testFolders))
{
  folderName = testFolders[testNumber];

  fileDirectory = paste(outputDirectory,"/",folderName,sep="");

  ## Import Group Quality
  groupQuality = read.csv(paste(fileDirectory,"/groupQualityData.csv",sep=""));
  extractGroupQuality = groupQuality$QallRelative;
  extractQ1 = groupQuality$Q1;
  extractQ2 = groupQuality$Q2;
  dataLength = length(extractGroupQuality);
  groupQualityVector[c(lineGroupQuality:(lineGroupQuality+dataLength-1))] = extractGroupQuality;
  Q1Vector[c(lineGroupQuality:(lineGroupQuality+dataLength-1))] = extractQ1;
  Q2Vector[c(lineGroupQuality:(lineGroupQuality+dataLength-1))] = extractQ2;
  lineGroupQuality = lineGroupQuality + dataLength;
}

### Output GroupQuality
groupQualityData = cbind(outputData[which(outputData$bee==1),],
                         Q1 = Q1Vector,
                         Q2 = Q2Vector,
                         groupQuality = groupQualityVector);
write.csv(groupQualityData,paste(outputDirectory,"/groupQualityData.csv",sep=""),row.names = F)







################Qnorm#########################

#bipartite function to compute the optimal Qnorm, from the paper 
# "Pasquaretta C, Jeanson R. Division of labor as a bipartite network. Behav Ecol. 2018;29(2): 342-352."

optim_matrix_bip <-function (net){
  ColSum<-sort(colSums(net),decreasing=T)
  RowSum<-sort(rowSums(net),decreasing=T)
  netMax<-matrix(0,nrow(net),ncol(net))
  while(sum(netMax)!=sum(net)){		
    Indiv <-which(ColSum-colSums(netMax)>0)
    Load.Indiv <-(ColSum-colSums(netMax))[ColSum-colSums(netMax)>0]
    Load.Task <-sort(rowSums(net), decreasing =T)-(rowSums(netMax))
    Load.Task[Load.Task <=0]<-NA
    MAT<-matrix(0, length(Load.Task),length(Indiv))
    rownames(MAT)<-1:length(Load.Task)
    colnames(MAT)<-which(ColSum-colSums(netMax)>0)
    MAT2<-MAT
    for (i in 1:length(Indiv)){
      MAT[,i]<-Load.Indiv[i]
      MAT2[,i]<-Load.Indiv[i]-Load.Task
    }
    tmp<-which(abs(MAT2)==min(abs(MAT2),na.rm=T),arr.ind=TRUE)[1,]
    if(MAT2[tmp[1],tmp[2]]<0)
      netMax[tmp[1], Indiv[tmp[2]]]<-Load.Indiv[tmp[2]]			
    if(MAT2[tmp[1],tmp[2]]>=0)
      netMax[tmp[1], Indiv[tmp[2]]]<-Load.Task[tmp[1]]					
  }
  return(netMax)
}

# now making the dataframe


for(testNumber in 1:length(testFolders))
{
  testFolderName = testFolders[testNumber];
  testFolderPath = paste(outputDirectory,"/",testFolderName,sep="");
  
  # Check if the file already exists
  testFiles = list.files(path = testFolderPath);
  if(any(testFiles=="H2Data.csv") & !overwriteFiles) 
  {
    cat("H2/Qnorm computation for test ",testFolderName," is already done. Proceeding to next test.\n",sep="");
    next;
  }
  
  cat("Starting H2/Qnorm assessment for test : ",testFolderName,".\n",sep="");
  
  # Initialize the output : one value of H2 per bout.
  output = data.frame(arrayNumber = rep(c(1:numberOfArrays),each=numberOfBouts*numberOfSimulations),
                      simulation = rep(c(1:numberOfSimulations),times=numberOfArrays,each=numberOfBouts),
                      bout = rep(c(1:numberOfBouts),times=numberOfArrays*numberOfSimulations),
                      H2 = 0,
                      Q = 0,
                      Qnorm = 0)
  i = 0;
  for(arrayNumber in 1:numberOfArrays)
  {
    arrayFolder = paste(testFolderPath,"/Array",sprintf("%02d",arrayNumber),sep="");
    
    bla = read.csv(paste(arrayFolder,"/arrayInfos.csv",sep=""))
    numberOfResources = bla$numberOfResources;
    
    # Import the visitation sequences of this array
    matrixOfVisitationSequences = as.matrix(unname(read.csv(paste(arrayFolder,"/matrixOfVisitationSequences.csv",sep=""))))
    
    for(sim in 1:numberOfSimulations)
    {
      # Isolate the visit sequences of the sim
      simVS = subset(matrixOfVisitationSequences,matrixOfVisitationSequences[,1]==sim);
      
      for(bout in 1:numberOfBouts)
      {
        i = i + 1;
        # Isolate the visit sequences of the bout
        boutVS = subset(simVS,simVS[,2]==bout);
        
        # Remove all 0 and 3 first columns.
        boutLength = apply(boutVS,2,sum); # Summing both lines. To get longest seq, we retrieve values > 0 (Smart).
        boutVS = boutVS[,boutLength>0];
        boutVS = boutVS[,-c(1:3)];
        
        # Count all visits
        flowerVisited = apply(boutVS,1,count);
        
        # Initialize the matrix at the correct format to be passed through the H2 assessment function.
        H2mat = matrix(0,nrow=numberOfResources,ncol=numberOfBees);
        
        # Fill the matrix with the counts
        for(bee in 1:numberOfBees)
        {
          
          visitCount = flowerVisited[[bee]];
          visitCount = visitCount[-which(visitCount$x==1 || visitCount$x==0),];
          
          for(flower in visitCount$x)
          {
            H2mat[flower-1,bee] = visitCount[which(visitCount$x==flower),2];
          }
        }
        
        # H2 computation
        output$H2[i] = H2fun(H2mat,H2_integer = T)[1];
        
        # Qnorm computation
        output$Q[i] = DIRT_LPA_wb_plus(H2mat)$modularity
        output$Qnorm[i] = DIRT_LPA_wb_plus(H2mat)$modularity / DIRT_LPA_wb_plus(optim_matrix_bip(H2mat))$modularity
        
        
        # Added part for Different Experience Sims. If no vector used by 2nd bee, H2 is 1.
        if(any(apply(H2mat,2,sum)==0)) { output[i,"H2"] = 1; output[i,"Q"] = 1; output[i,"Qnorm"] = 1}
      }
    }
  }
  write.csv2(output,paste(testFolderPath,"/H2Data.csv",sep=""),row.names=F)
}

# Join the files
H2Vector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts)
QVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts)
QnormVector = numeric(numberOfTests*numberOfArrays*numberOfSimulations*numberOfBouts)
lineH2 = 1;

for(testNumber in 1:length(testFolders))
{
  folderName = testFolders[testNumber];
  cat("Importing data from :",folderName,"\n");

  fileDirectory = paste(outputDirectory,"/",folderName,sep="");

  ## Import H2
  H2Data = read.csv2(paste(fileDirectory,"/H2Data.csv",sep=""));
  colnames(H2Data) = c("arrayNumber","simulation","bout","H2","Q","Qnorm");

  extractH2 = H2Data$H2;
  extractQ = H2Data$Q;
  extractQnorm = H2Data$Qnorm;
  dataLength = length(extractH2);
  H2Vector[c(lineH2:(lineH2+dataLength-1))] = extractH2;
  QVector[c(lineH2:(lineH2+dataLength-1))] = extractQ;
  QnormVector[c(lineH2:(lineH2+dataLength-1))] = extractQnorm;
  lineH2 = lineH2 + dataLength;
}

H2Data = cbind(outputData[which(outputData$bee==1),],H2 = H2Vector,Q = QVector, Qnorm = QnormVector);
write.csv(H2Data,paste(outputDirectory,"/H2Data.csv",sep=""),row.names = F)







############plots###################


#competition interference
data=read.csv(paste(outputDirectory,"/competitionData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[data$learningValue==1.5 & data$abandonValue==1.]="model 1"
modelVector[data$learningValue==1. & data$abandonValue==0.75]="model 2"
modelVector[data$learningValue==1.5 & data$abandonValue==0.75]="model 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data=aggregate(data,list(data$arrayType,data$model,data$bout),mean)
colnames(data)[2]="Model"
ggplot(data = data,aes(x=bout, y=compInterference,group=Model,color=Model)) +
  geom_line()+
  facet_wrap(~Group.1)

#competition exploitation
data=read.csv(paste(outputDirectory,"/competitionData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[data$learningValue==1.5 & data$abandonValue==1.]="model 1"
modelVector[data$learningValue==1. & data$abandonValue==0.75]="model 2"
modelVector[data$learningValue==1.5 & data$abandonValue==0.75]="model 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data=aggregate(data,list(data$arrayType,data$model,data$bout),mean)
colnames(data)[2]="Model"
ggplot(data = data,aes(x=bout, y=compExploitation,group=Model,color=Model)) +
  geom_line()+
  facet_wrap(~Group.1)

#similarity index
data=read.csv(paste(outputDirectory,"/similarityData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[data$learningValue==1.5 & data$abandonValue==1.]="model 1"
modelVector[data$learningValue==1. & data$abandonValue==0.75]="model 2"
modelVector[data$learningValue==1.5 & data$abandonValue==0.75]="model 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data=aggregate(data,list(data$arrayType,data$model,data$bout),mean)
colnames(data)[2]="Model"
ggplot(data = data,aes(x=bout, y=similarityIndex,group=Model,color=Model)) +
  geom_line()+
  facet_wrap(~Group.1)+
  ylim(0,1)

#group quality
data=read.csv(paste(outputDirectory,"/groupQualityData.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[data$learningValue==1.5 & data$abandonValue==1.]="model 1"
modelVector[data$learningValue==1. & data$abandonValue==0.75]="model 2"
modelVector[data$learningValue==1.5 & data$abandonValue==0.75]="model 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data=aggregate(data,list(data$arrayType,data$model,data$bout),mean)
colnames(data)[2]="Model"
ggplot(data = data,aes(x=bout, y=groupQuality,group=Model,color=Model)) +
  geom_line()+
  facet_wrap(~Group.1)+
  ylim(0,1)

#Qnorm
data=read.csv(paste(outputDirectory,"/H2Data.csv",sep=""))
modelVector=rep('None',nrow(data))
modelVector[data$learningValue==1.5 & data$abandonValue==1.]="model 1"
modelVector[data$learningValue==1. & data$abandonValue==0.75]="model 2"
modelVector[data$learningValue==1.5 & data$abandonValue==0.75]="model 3"
data["model"]=modelVector
nArrayTypes=length(levels(as.factor(data$arrayType)))
data=aggregate(data,list(data$arrayType,data$model,data$bout),mean)
colnames(data)[2]="Model"
ggplot(data = data,aes(x=bout, y=Qnorm,group=Model,color=Model)) +
      geom_line()+
      facet_wrap(~Group.1)+
      ylim(0,1)
