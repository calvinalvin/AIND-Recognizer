import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    where ~ L is the likelihood of the fitted model
          ~ p is the number of parameters
          ~ N is the number of data points
    """
    
    def bic_model(self, num_states):
        """ gets the BIC score for a model given the number HMM hidden states
        parameters =  n_components*n_components + 2*n_components*n_features - 1
        logL = word_score
        logN = log(number_of_data_points)
        BIC = -2 * logL + parameters * logN
        """
        
        hmm_model = self.base_model(num_states)
        
        # Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
        # ~ Initial state occupation probabilities = numStates
        # ~ Transition probabilities = numStates*(numStates - 1)
        # ~ Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars
        p = num_states*num_states + 2*num_states*len(self.X[0])-1
        logL = hmm_model.score(self.X, self.lengths)
        logN = math.log(len(self.X))
        bic = -2 * logL + p * logN
        return bic, hmm_model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            best_num_states = 1
            best_bic_score = float("-inf")
            best_model = self.base_model(self.n_constant)
            
            for n in range(self.min_n_components, self.max_n_components):
                try:
                    score, model = self.bic_model(n)
                    if score > best_bic_score:
                        best_bic_score = score
                        best_num_states = n
                        best_model = model
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
                    pass
              
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, best_num_states))
            return best_model
        
        except:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    where ~ log(P(X(i)) is the probability of the evidence or model score
          ~ log(P(X(all but i)) is the probability of the anti evidence or prob of all the words except the word you're studying
    '''

    def anti_dic_score(self, hmm_model):
        """ gets the average model score calculation for all words except this_word """
        total_score = 0
        word_count = 0
        avg_score = 0
        
        for word, features in self.hwords.items():
            if self.this_word != word:
                X, lengths = features
                score = hmm_model.score(X, lengths)
                total_score += score
                word_count += 1
                    
        avg_score = total_score / word_count
        return avg_score
                
    
    def dic_model(self, num_states):
        """determines the DIC score for the given number of states"""
        
        hmm_model = self.base_model(num_states)
        
        # log(P(X(i))
        p_evidence = hmm_model.score(self.X, self.lengths)
        # 1/(M-1)SUM(log(P(X(all but i))
        p_anti_evidence = self.anti_dic_score(hmm_model)
        dic_score = p_evidence - p_anti_evidence
        
        return dic_score, hmm_model
        
    
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try: 
            best_num_states = 1
            best_dic_score = float("-inf")
            best_model = self.base_model(self.n_constant)
                    
            # implement model selection based on DIC scores
            for n in range(self.min_n_components, self.max_n_components):
                try:
                    score, model = self.dic_model(n)
                    if score > best_dic_score:
                        best_dic_score = score
                        best_model = model
                        best_num_states = n

                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
                    pass
                
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, best_num_states))
            return best_model
         
        except:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cross_validation_model(self, num_states, n_splits=3):
        """ builds a model using a kfolds split data set against training data/ test data to avoid overfitting
        """
        total_score = 0
        
        if(len(self.sequences) > n_splits):
            split_method = KFold(n_splits=n_splits)
            folds = 0
            
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                folds +=1
                # split the data into training and test validation data
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                 
                # train on just the train data
                training_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                # then test against the test data
                score = training_model.score(X_test, lengths_test)
                total_score += score
             
            # return the average of the all the scores
            avg_score = total_score/folds
            return avg_score, training_model
        
        else:
            return 0., self.base_model(self.n_constant)
        
    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_num_states = 0
        best_score = float("-inf")
        best_model = None

        try:
            for n in range(self.min_n_components, self.max_n_components):
                score, model = self.cross_validation_model(n)
                if score >= best_score:
                    best_score = score
                    best_model = model
                    best_num_state = n
                    
            return best_model

        except Exception as e: 
            print(e)
            return None
            