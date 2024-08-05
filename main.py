from src.data.make_dataset import load_data, clean_and_classify_data, univariate_analysis, data_prep_model
from src.visualization.visualize import plot_histograms, plot_attrition_vs_categorical,plot_correlation_heatmap
from src.features.build_features import univariate_categorical_analysis,relationship_attrition_numerical
from src.models.train_model import train_and_evaluate_models

#Data Path Definition
if __name__ == "__main__":
    # Load Data
    data_path = r"F:\\BISI\\Data Science Project\\Employee_Attrition_using_SVM\\data\\raw\\HR_Employee_Attrition.xlsx"
    df = load_data(data_path)

# Check unique values in each column # Check unique values in each column
    print("Unique values in each column:")
    print(df.nunique())
    
# Clean Data and classify columns
    df, num_cols, cat_cols = clean_and_classify_data(df)
    
# Univariate analysis of numerical columns
univariate_analysis(df,num_cols)

# Plot histograms for numerical columns
plot_histograms(df, num_cols)


# Univariate analysis for categorical variables
univariate_categorical_analysis(df, cat_cols)

# Bivariate analysis: Attrition vs Categorical variables
plot_attrition_vs_categorical(df, cat_cols)

# Relationship between attrition and numerical variables
relationship_attrition_numerical(df, num_cols)

# Plot correlation heatmap
plot_correlation_heatmap(df, num_cols)

# Data Preparation for Modeling
x_train, x_test, y_train, y_test = data_prep_model(df)
print(df.describe())


# Train and evaluate models
train_and_evaluate_models(x_train, x_test, y_train, y_test)