# include the required libraries
library(RTextTools)
library(e1071)
library(tm)
library(caret)
library(plotrix)




# reading data into data frames
reviews = read.csv("CleanedCombinedDataCompressed.csv", header = FALSE, fill = TRUE)
augmented.reviews = read.csv("AugmentedTrainingData.csv", header = FALSE, fill = TRUE)


lbls <- unique(reviews$V2)
slices <- c(nrow(reviews[reviews$V2 == lbls[1], ]), 
            nrow(reviews[reviews$V2 == lbls[2], ]), 
            nrow(reviews[reviews$V2 == lbls[3], ]))
pie3D(slices, labels = lbls, explode = 0.1, main = "Composition of Training Data")
##############################################################################################################################

# build dtm
mat= create_matrix(reviews[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=TRUE) 
sparse <- removeSparseTerms(mat, 0.7)
matrix = as.matrix(sparse)

aug.mat= create_matrix(augmented.reviews[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=TRUE) 
                                                                                    sparse <- removeSparseTerms(aug.mat, 0.81)
aug.matrix = as.matrix(sparse)
augmented.matrix = cbind(aug.matrix[,], augmented.reviews[,2], augmented.reviews[,3])
#############################################################################################


# Naive Bayes Technique

classifier = naiveBayes(matrix[1:800,], as.factor(reviews[1:800,2]),laplace = 1 )
predicted = predict(classifier, matrix[801:1192,])
predicted.table = table(reviews[801:1192, 2], predicted)
predicted.table
plot(predicted.table)
############################################################################################
recall_accuracy(reviews[801:1192, 2], predicted)
conf.mat <- confusionMatrix(predicted,reviews[801:1192,2])
conf.mat$overall["Accuracy"]
naive.bayes.accuracy = as.numeric(conf.mat$overall["Accuracy"])
########################################################################################


classifier = naiveBayes(augmented.matrix[1:800,], as.factor(augmented.reviews[1:800,4]),laplace = 1 )
predicted = predict(classifier, augmented.matrix[801:1192,])
predicted.table = table(augmented.reviews[801:1192, 4], predicted)
predicted.table
plot(predicted.table)
############################################################################################
recall_accuracy(augmented.reviews[801:1192, 4], predicted)
conf.mat <- confusionMatrix(predicted,augmented.reviews[801:1192,4])
conf.mat$overall["Accuracy"]
naive.bayes.hybrid.accuracy = as.numeric(conf.mat$overall["Accuracy"])
############################################################################################


# build container for trsining models and training models
container = create_container(matrix, as.numeric(as.factor(reviews[,2])),
                             trainSize=1:800, testSize=801:1192,virgin=FALSE)
models = train_models(container, algorithms=c("SVM", "RF"))
results = classify_models(container, models)

                                                                                          #sparse <- removeSparseTerms(aug.mat, 0.7); aug.matrix = as.matrix(sparse); augmented.matrix = cbind(aug.matrix[,], augmented.reviews[,2], augmented.reviews[,3])
augmented.container = create_container(augmented.matrix, as.numeric(as.factor(augmented.reviews[,4])),
                             trainSize=1:800, testSize=801:1192,virgin=FALSE)
augmented.models = train_models(augmented.container, algorithms=c("SVM", "RF"))
augmented.results = classify_models(augmented.container, augmented.models)
############################################################################################


# Support Vector Machine Technique
table(as.factor(reviews[801:1192, 2]), results[,"SVM_LABEL"])
N=4
set.seed(2014)
svm.accuracy=cross_validate(container,N,"SVM")$meanAccuracy
svm.accuracy
svm.accuracy=as.numeric(svm.accuracy)
#svm.accuracy = as.numeric(cross_validate(container,N,"SVM"))

table(as.factor(augmented.reviews[801:1192, 4]), augmented.results[,"SVM_LABEL"])
N=4
set.seed(2014)
svm.hybrid.accuracy=cross_validate(augmented.container,N,"SVM")$meanAccuracy
svm.hybrid.accuracy
svm.hybrid.accuracy=as.numeric(svm.hybrid.accuracy)
#svm.hybrid.accuracy = as.numeric(cross_validate(augmented.container,N,"SVM"))
############################################################################################

# Random Forest Technique
table(as.factor(reviews[801:1192, 2]), results[,"FORESTS_LABEL"])
N=4
set.seed(2014)
rf.accuracy=cross_validate(container,N,"RF")$meanAccuracy
rf.accuracy
rf.accuracy=as.numeric(rf.accuracy)
#rf.accuracy = as.numeric(cross_validate(container,N,"RF"))

table(as.factor(augmented.reviews[801:1192, 4]), augmented.results[,"FORESTS_LABEL"])
N=4
set.seed(2014)
rf.hybrid.accuracy=cross_validate(augmented.container,N,"RF")$meanAccuracy
rf.hybrid.accuracy
rf.hybrid.accuracy = as.numeric(rf.hybrid.accuracy)
#rf.hybrid.accurac = as.numeric(cross_validate(augmented.container,N,"RF"))
############################################################################################


# Comparison Graph for Accuracy of Various Techniques
counts <- rbind( c(naive.bayes.accuracy, svm.accuracy, rf.accuracy), 
                 c(naive.bayes.hybrid.accuracy, svm.hybrid.accuracy, rf.hybrid.accuracy))
barplot(counts, col = c("darkblue", "green"), beside = TRUE)
############################################################################################

# Naive Bayes Revisited without Neutral Class
reviews = read.csv("CleanedCombinedDataWithoutNeutralCompressed.csv", header = FALSE, fill = TRUE)
matrix= create_matrix(reviews[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=TRUE) 
sparse <- removeSparseTerms(matrix, 0.6)
mat = as.matrix(sparse)
classifier = naiveBayes(mat[1:600,], as.factor(reviews[1:600,2]),laplace = 1 )
predicted = predict(classifier, mat[601:791,])
table(reviews[601:791, 2], predicted)
recall_accuracy(reviews[601:791, 2], predicted)
conf.mat <- confusionMatrix(predicted,reviews[601:791,2])
conf.mat$overall["Accuracy"]
naive.bayes.accuracy = as.numeric(conf.mat$overall["Accuracy"])

augmented.reviews = read.csv("AugmentedTrainingDataWithoutNeutral.csv", header = FALSE, fill = TRUE)
aug.mat= create_matrix(augmented.reviews[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=TRUE) 
sparse <- removeSparseTerms(aug.mat, 0.7)
aug.matrix = as.matrix(sparse)
augmented.matrix = cbind(aug.matrix[,], augmented.reviews[,2], augmented.reviews[,3])
classifier = naiveBayes(augmented.matrix[1:600,], as.factor(augmented.reviews[1:600,4]),laplace = 1 )
predicted = predict(classifier, augmented.matrix[601:791,])
table(augmented.reviews[601:791, 4], predicted)
recall_accuracy(augmented.reviews[601:791, 4], predicted)
conf.mat <- confusionMatrix(predicted,augmented.reviews[601:791,4])
conf.mat$overall["Accuracy"]
naive.bayes.hybrid.accuracy = as.numeric(conf.mat$overall["Accuracy"])

counts <- rbind(naive.bayes.accuracy, naive.bayes.hybrid.accuracy)
barplot(counts, col = c("darkblue", "green"),beside = TRUE)

