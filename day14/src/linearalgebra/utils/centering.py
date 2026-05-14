#create data centering for trainee score dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from linearalgebra.configurations.conf import Config

def center_data(file_path):
    #read the dataset
    df = pd.read_csv(file_path)
    #compute the center
    #calculate the mean of the score column
    mean_score = df['score'].mean()
    #x_centered = x - mean_score
    #center the data
    df['score_centered'] = df['score'] - mean_score
    return df
def create_plot(df):
    import matplotlib.pyplot as plt
    plt.hist(df['score_centered'], bins=20)
    plt.title('Centered Scores Distribution')
    plt.xlabel('Centered Score')
    plt.ylabel('Frequency')
    plt.show()
if __name__ == "__main__":
    file_path = Config.score_path
    centered_df = center_data(file_path)
    print(centered_df.head())
    create_plot(centered_df)
