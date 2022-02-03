
#NEED TO FINISH THIS
def choose_func(name,df,vars):

    if name == 'selfpaced_threshold':
        return selfpaced_thresholded(df,vars)
    elif name == 'hardfirst':
        return #ADD
    elif name == 'normal':
        return resample_normal(df)
    
        
#All the acquisition functions used 

def selfpaced_thresholded(df,vars):
    '''
    increase the amount of data availible based on an increasing threshold by lambda amount
    record the amount of data used in 'dataused'
    '''
    df = df[df['diff'] < vars.thresh]
    vars.thresh = vars.thresh * vars.lam
    vars.dataused.append(len(df.index))
    return df, vars

def threshold_online_variance(df,vars):
    '''
    n epoch hold out period 
    Reducing the threshold by lambda
    calculating the variance based on the last 3 loss results (online)
    '''
    #resampling set amount of data
    #calculate the variance
    if vars.epoch <= 3:
        df = df.sample(frac=1,replace=False)
    else:
        df['var'] = df[['l1','l2','l3']].var(axis=1)
        df = df[df['var'] > vars.thresh]
        vars.thresh = vars.thresh / vars.lam

    return df

def resample_dataset(df,epoch):
    #selects more difficult data more often based on diff
    #T1
    #np_diffs = df['diff'].to_numpy()
    #normalise the diffs to sum to 1 (techniqualy not needed as df.sample normalises if weights does not sum = 1)
    #s = sum(np_diffs)
    #norm_diffs = [float(i)/s for i in np_diffs]

    #resampling set amount of data
    #df = df.sample(frac=1, weights=norm_diffs, replace=True)

    #T2
    if epoch % 2 == 0:
        df = df.sample(frac=1,replace=False)
    else:
        #resampling set amount of data
        df = df.sample(frac=1, weights=df['diff'], replace=True)

    #T3
    #if epoch % 10 == 0:
    #    df = df.sample(frac=1,replace=False)
    #else:
    #    #sample but exclude the outliers based on 1.5std
    #    standard_dev = df['diff'].std()
    #    np_diffs = df['diff'].apply(lambda x: 0 if x > (1.5*standard_dev)  else x)
    #    df = df.sample(frac=1,replace=True)

    return df

def resample_normal_and_diff(df,percentage_Resample):
    #Percentage normal is [0 to 1]
    batchSize = 32
    numSampled = math.floor(batchSize*percentage_Resample)
    #Create the entire epoch randomly
    new_df = df.sample(frac=1,replace=False).reset_index(drop=True)
    #Replace the difficulty sampled section every batch
    for i in range(0,len(df)-batchSize,batchSize):
        new_df.iloc[i:i+numSampled] = df.sample(n=numSampled, weights=df['diff'], replace=True).values
    return new_df

def resample_normal(df):
    df = df.sample(frac=1,replace=False)
    return df

def resample_hardbatches(df):
    #for each batch add multiples of hard data and have perfect mixing
    df = df.sample(frac=1,replace=False) #shuffle

    #add more hard data
    hard_thresh = int(len(df.index)*0.5)

    harddf = df.nlargest(hard_thresh,'diff')

    df = pd.concat([df,harddf]) #now with extra hard data

    easy_index = np.linspace(0,len(df.index)-1,len(df.index)) #
    hard_index = np.linspace(len(df.index)-1,0,len(df.index))

    

    lis = []
    for i in range(0,math.ceil(len(df.index)/2)):
        if i == math.ceil(len(df.index)/2) & len(df.index)%2 != 0: #if the last value is not divisible by 2 only append one number
            lis.append(easy_index[i])
            break
        lis.append(easy_index[i])
        lis.append(hard_index[i])

    #reformat so the data goes easy hard easy hard ...
    df = df.sort_values(by=['diff']).reset_index(drop=True)
    df = df.reindex(lis).reset_index(drop=True)
    return df

def resample_dataset_easy_first(df):
    #resample the data so during an epoch easy imgs are clumped together
    df = df.sort_values(by=['diff'])
    return df

def resample_dataset_hard_first(df):
    df = df.sort_values(by=['diff'],ascending=False)
    return df

def resample_dataset_easy_hard_mix(df):
    #ensure each batch has similar easy and hard data
    #example batch 1 contains [hardest image, easiest image, 2nd hardest img ...]
    #Create index values
    df_new = df.copy()
    easy_index = np.linspace(0,len(df.index)-1,len(df.index)) #
    hard_index = np.linspace(len(df.index)-1,0,len(df.index))

    lis = []
    for i in range(0,math.ceil(len(df.index)/2)):
        if i == math.ceil(len(df.index)/2) & len(df.index)%2 != 0: #if the last value is not divisible by 2 only append one number
            lis.append(easy_index[i])
            break
        lis.append(easy_index[i])
        lis.append(hard_index[i])

    #reformat so the data goes easy hard easy hard ...
    df_new = df_new.sort_values(by=['diff']).reset_index(drop=True)
    df_new = df_new.reindex(lis).reset_index(drop=True)
    
    #removing data part
    #df_new = df_new.nlargest(25000)

    return df_new