#create bernoulli distribution using email spam dataset from resources folder
import pandas as pd
from scipy.stats import bernoulli
from distapp.configurations.config import Config    

def create_bernoulli_dist():
    config = Config()
    df = pd.read_csv(config.bernoulli_path)
    #create bernoulli distribution using email spam dataset from resources folder
    #calculate the probability of spam emails in the dataset using label column
    # label column has text either spam or not spam, so we can calculate the probability of spam emails by counting the number of spam emails and dividing by the total number of emails
    p = df['label'].value_counts()['spam'] / len(df)   
    print(f"Probability of spam emails: {p}")
    print(f"Probability of not spam emails: {1 - p}")
    #create a bernoulli distribution object using the calculated probability
    bernoulli_dist = bernoulli(p)
    return bernoulli_dist

if __name__ == "__main__":
    bernoulli_dist = create_bernoulli_dist()
    #generate 25 random samples from the bernoulli distribution
    samples = bernoulli_dist.rvs(size=25)
    print(f"Random samples from the bernoulli distribution: {samples}")
