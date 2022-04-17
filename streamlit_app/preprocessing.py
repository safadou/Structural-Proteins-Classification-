
from curses import def_prog_mode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from scipy.stats import skew


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Protein Data Preprocessing class
    The class is built as transformer to be used in a pipeline
    
    Attributes
    ----------
    verbose:

    Methods
    -------
    

    """
    def __init__(self, verbose=True):
      super().__init__()
      self.verbose = verbose

    def fit(self, X, y=None):  
        #This is done at fit time, to have the expected columns to be applied when predicting
        #When exporting the model and testing with new data, the predict complains when some columns unseen
        #during fit columns appears
        #Maybe there is a way to do differently?
        self.fit_data_in = X
        to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear', 'crystallizationMethod','experimentalTechnique']
        X = X.drop([x for x in to_drop if x in X.columns], axis=1)
        X = self.handle_missing(X)
        X = self.reduce_modalities(X)
        X = self.handle_skewness(X)
        X = self.scale_encode_data(X)
        if self.verbose : print("-- Fit done -- ")
        self.fitted_columns = X.columns
        self.fit_data_out = X
        return self
    
    def transform(self, X) :
        verbose = self.verbose
        #If transforming the fitted array, just return the fit_data_out
        if np.array_equal(X, self.fit_data_in) :
            return self.fit_data_out

        if verbose : print("1.Drop useless columns")
        to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear', 'crystallizationMethod','experimentalTechnique']
        X = X.drop([x for x in to_drop if x in X.columns], axis=1)

        if verbose : print('2.Replace missing values in X')
        X = self.handle_missing(X)

        if verbose : print('3.Reduce modalities')
        X = self.reduce_modalities(X)

        if verbose : print("4.Correct skewness")
        X = self.handle_skewness(X)

        if verbose : print("5.scale and encode categ values")
        X = self.scale_encode_data(X)
        
        if verbose : print("-- Preprocessing done -- ")

        #Check the columns in the 
        nb_cols_in  = set(X.columns)
        nb_cols_fit = set(self.fitted_columns)  
        if nb_cols_in != nb_cols_fit :
            print("Not All fitted values seen")
            #Add the missing columns in case
            X = self.add_missing_cols(X)
            #This will restrict the output to the fitted data expected by the model
            X = X[self.fitted_columns]
        if verbose: print("[Columns] : ",self.fitted_columns)
        return X

    ######
    ######  Methods 
    ######

    def add_missing_cols(self, X):
        X = X.copy()
        cols_added=[]
        for col in self.fitted_columns:
            if col not in X:
                print("Adding column [",col,"]")
                X[col] = 0
                cols_added.append(col)
        return X


    def handle_missing(self, df):
        data = df.copy()
        for i in df.select_dtypes(exclude = np.number):
            data[i] = data[i].fillna(data[i].mode()[0])
        # for numerical values replace nan by median 
        data = data.fillna(df.select_dtypes(include = np.number).median())
        # drop anything else (useless)
        data = data.dropna()
        return data.reset_index(drop=True)


    def reduce_modalities(self, df):
        df = df.copy()
        # analyse all variables separately
        #df['experimentalTechnique'] = self.__map_experimentalTechique(df.experimentalTechnique)
        #df['crystallizationMethod'] = self.__map_crystallizationMethod(df.crystallizationMethod)
        df['macromoleculeType']     = self.map_macromoleculeType(df.macromoleculeType)
        df['phValue']               = self.map_phValue(df.phValue)
        return df

    def map_experimentalTechique(self, data):
        expt = (data.value_counts(normalize=True))
        techniques={}
        for tech in expt.index:
            if 'X-RAY DIFFRACTION' in tech:
                techniques[tech] = 'X-RAY DIFFRACTION' 
            else:
                techniques[tech] = 'others_tech_exp' 
        # return new experimentalTechniques values in dataframe
        return data.map(techniques)

    def map_crystallizationMethod(self, data):
        data = data.str.replace(',', ' ')
        data = data.str.replace('-', ' ')
        data = data.str.lower()

        cm = (data.value_counts(ascending = False, normalize=True) *100)

        methods={}

        for method, value in zip(cm.index, cm.values) : 
            if method.startswith('vapor diffusion'):
                methods[method] = 'vapor diffusion'

            elif method.startswith('microbatch'):
                methods[method] = 'microbatch'

            elif method.startswith('batch'):
                methods[method] = 'batch'

            elif method.startswith('evaporation'):
                methods[method] = 'evaporation'

            elif method.startswith('sitting drop'):
                methods[method] = 'sitting drop'
                
        return data.map(methods)
        
    def map_macromoleculeType(self, data):
      mt = (data.value_counts(normalize=True)*100)
      # Les 3 premiers types constituent 97% des modalités
      # Keep the 3 first and regroup the rest in OTHERS
      mTypes={}
      for m, value in zip(mt.index, mt.values):
        if m.startswith('Protein'):
            mTypes[m] = 'Protein'
        elif m.startswith('DNA'):
            mTypes[m] = 'DNA'
        elif m.startswith('RNA'):
            mTypes[m] = 'RNA'
        else:
            mTypes[m] = "others_macro_mol"

      return data.map(mTypes)

    @staticmethod
    def __ph_value(ph):
      if ph < 7.0:
          return 'acide'
      elif ph > 7.0:
          return 'basique'
      else:
          return 'neutre'

    def map_phValue(self, data):
        #data.apply(PreprocessingTransformer.__ph_value, args=(self))
	    return data.map(lambda p: PreprocessingTransformer.__ph_value(p))

    def handle_skewness(self, data) :
        # calculate skewness for all numerical values in dataframe
        num_features = data.select_dtypes(include=np.number)
        tmp = num_features
        for i in tmp:
            if abs(skew(num_features[i])) > 1 :
                #Replace data which has skewness > 1 by log(data)
                num_features[i]=tmp[i].map(lambda x: np.log(x) if x > 0 else 0)
        df = data
        tmp_f = num_features
        if len(num_features.columns) != len(df.columns):
            #replace only num features in the provided dataset
            for i in tmp_f:
                df[i] = num_features[i]
        else:
            df =  num_features   
        return df

    def scale_encode_data(self, data):
        df = data.copy()
        num_data = PreprocessingTransformer.scaleData(df.select_dtypes(include = "number"))
        for col in num_data:
            df[col] = num_data[col]
        
        cat_data = df.select_dtypes(include = object)
        df = pd.get_dummies(df, prefix_sep= '_', drop_first=False, columns=cat_data.columns)
        return df

    @staticmethod
    def scaleData(num_data):
        scaler = RobustScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(num_data), columns = num_data.columns)
        return data_scaled.reset_index(drop=True)


    @staticmethod
    def filter_classification(df, threshold=5000):
        """
        Removes all classes with count values < threshold

        Parameters
        ----------
        df : pd.DataFrame
            complete dataframe
        
        threshold : int
            minimum amount to keep

        """
        counts = df.classification.value_counts()
        types = np.asarray(counts[counts > threshold].index)
        return df[df.classification.isin(types)].copy()

    @staticmethod
    def map_classification(data, threshold):
        counts  = data.value_counts()
        up_thresh_indexes = np.asarray(counts[counts > threshold].index)
        classes={}
        for c in data:
            if c in up_thresh_indexes : 
                classes[c] = c
            else :
                classes[c]='other_classes'
        
        return data.map(classes)

def load_datasets():
    # Open file 1
    df_prot = pd.read_csv('../data/data_no_dups.csv')
    # Open file 2
    df_seq = pd.read_csv('../data/data_seq.csv')
    # merge files 
    df = pd.merge(df_prot, df_seq, on =['structureId','macromoleculeType', 'residueCount' ],   how = 'inner')
    # return merged data
    return df

def format_classification(df):
    df['classification'] = df['classification'].str.lower()
    df['classification'] = df['classification'].str.strip()
    df['classification'] = df['classification'].str.replace('(','/', regex=False) 
    df['classification'] = df['classification'].str.replace(',','/', regex=False) 
    df['classification'] = df['classification'].str.replace(', ','/', regex=False) 
    df['classification'] = df['classification'].str.replace('/ ','/', regex=False) 
    df['classification'] = df['classification'].str.replace(')','', regex=False)  

    classes = df.classification.value_counts().index

    composed = []
    single = []
    for i in classes :
        if '/' in i.strip() :
            composed.append(i)
        else:
            single.append(i) 

    new_c={}
    #i = 0
    for s in single:
        for c in classes:
            #if s in c and c not in new_c:
            #    new_c[c] = s
            if c not in new_c:
                new_c[c] = c
                
            if c.startswith(s):
                new_c[c] = s
                #if i < 10 : print(c, new_c[c])
                #i += 1

    df['classification'] = df.classification.map(new_c)

    return df

def prepare_target(df, strategy='filter', threshold=5000):
    """
    Removes NaN from Target
    Reduce modalities by filtering the values in the target 
    """
    if df.classification.isna().sum() > 0 : 
        #Remove NaN on target => put mode value
        df['classification'] = df.classification.fillna(df.classification.mode()[0])

    print(f'\033[1mComplete DataFrame has {df.shape[0]} lines and {df.shape[1]} columns')

    df = format_classification(df)

    if strategy == 'regroup':
        df_group = df.copy()
        df_group['classification'] = PreprocessingTransformer.map_classification(df_group.classification, threshold)
        print(f'\033[1mFinal DataFrame has {df_group.shape[0]} lines and {df_group.shape[1]} columns after regrouping all classes with less than {threshold} items')
        return df_group
    else: 
        #'remove filter, or None
        #Filter to a threshold of 5000 values
        df_filtered = PreprocessingTransformer.filter_classification(df, threshold)
        print(f'\033[1mFinal DataFrame has {df_filtered.shape[0]} lines and {df_filtered.shape[1]} columns after removing all classes with less than {threshold} items')
        return df_filtered