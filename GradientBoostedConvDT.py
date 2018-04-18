
from ConvDT_gpu import ConvDT

###########################################
### Implementation of Gradient Boosting ###
###########################################

from sklearn.ensemble.gradient_boosting as GB

class GradientBoostedConvDT(base_ConvDT):

    def __init__(self, num_estimators, learning_rate):
        self.num_estimators = num_estimators
        self.learning_rate = learning_rate
        self.loss_ = GB.BinomialDeviance(2)
        self.base_ConvDT = base_ConvDT

    def fit(self, X, y):
        self.estimators_ = []

        ## Get initial constant estimate ##
        self.init_estimator = self.loss_.init_estimator()
        self.init_estimator.fit(X,y)
        y_pred = self.init_estimator.fit(X,y)

        ## Start it off by getting the first residuals ##
        residuals = self.loss_.negative_gradient(y, y_pred)
        
        ## Train the ConvDTs ##
        for i in range(self.num_estimators):
            ## create the ConvDT and fit to residuals##
            estimator = copy.deepcopy(self.base_ConvDT)
            estimator.fit(X,residuals)

            ## Add the ConvDT to list of esimators ##
            self.estimators_.append(estimator)

            ## update the prediction ##
            y_pred = self._decision_function_gradual(X, y_pred)

            ## Get new residuals ##
            residuals = self.loss_.negative_gradient(y, y_pred)

        return self

    def _decision_function_gradual(X, y_current):
        predicted_probs = self.estimators_[-1].predict_proba(X)

        return y_current + (self.learning_rate * np.log(predicted_probs[:,0] / predicted_probs[:,1]))

    def decision_function(X):
        
        output = self.init_estimator(X)

        for estimator in self.estimators_:
            predicted_probs = estimator.predict_proba(
            output += self.learning_rate * np.log(predicted_probs[:,0] / predicted_probs[:,1])

        return output
