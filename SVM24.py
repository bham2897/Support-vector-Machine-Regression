import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

print("Start processing with SVM Regressor and Permutation Importance:")

# Load the data
file_path = '/Users/divya/Downloads/for_python.xlsx'  
data = pd.read_excel(file_path)
print("Data loaded successfully.")

# Convert datetime columns to numerical values
data['Attendance Detail Date'] = pd.to_datetime(data['Attendance Detail Date'])
data['First Entry Date'] = pd.to_datetime(data['First Entry Date'])
print("Datetime columns converted to pandas datetime format.")

# Calculate 'Frequency of Entry' as the count of each Unique Key
data['Frequency of Entry'] = data.groupby('Unique Key')['Unique Key'].transform('count')
print("Frequency of Entry calculated.")

# Extract month name from 'First Entry Date' and encode it
data['First Entry Month'] = data['First Entry Date'].dt.strftime('%B')  # Convert to month name (e.g., "October")
print("Month extracted from 'First Entry Date'.")

# Drop original datetime columns if they are not needed
data = data.drop(columns=['Attendance Detail Date', 'First Entry Date'])
print("Dropped original datetime columns.")

# Encode categorical variables
categorical_cols = ['Entry Time Category', 'Attendance Detail Weekday', 'Contacts Detail Gender', 'Contacts Detail Price Level', 'Membership Level', 'First Entry Month']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the label encoder for potential inverse transformation
print("Categorical columns encoded.")

# Features used for prediction 
X = data[['Contacts Detail Age', 'Contacts Detail Gender', 'Attendance Detail Weekday', 'Membership Level', 'Contacts Detail Price Level']]

# Modify feature names to include line breaks for better visibility
X.columns = [
    'Contacts Detail\nAge',
    'Attendance Detail\nWeekday',
    'Membership\nLevel',
    'Contacts Detail\nPrice Level',
    'Contacts Detail\nGender'
]
print("Features selected and column names modified for visibility.")

# List of activities to predict
activities = ['Gym Accessed', 'Pool Accessed', 'Classes', 'Other Activities', 'Alternative Sessions']

# Initialize lists to collect overall scores and predicted vs actual data
overall_r2_scores_svr = []
overall_mse_scores_svr = []
overall_rmse_scores_svr = []
combined_results_df_svr = pd.DataFrame()

# Model by default has used RBF kernel 
models_svr = {}
feature_importances_svr = {}

for activity in activities:
    print(f"Processing activity: {activity}")
    
    # Target variable (binary encoded)
    y = data[activity]
    
    # Split the data into training and testing sets for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    print("Data split into training and testing sets.")
    
    # Initialize the SVM Regressor
    model_svr = SVR()
    
    # Train the model
    model_svr.fit(X_train, y_train)
    print(f"SVM model trained for {activity}.")
    
    # Store the model for later use in predictions
    models_svr[activity] = model_svr
    
    # Make predictions
    y_pred_svr = model_svr.predict(X_test)
    print(f"Predictions made for {activity}.")
    
    # Evaluate the regression model
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"Model evaluation completed for {activity}. R²: {r2_svr}, MSE: {mse_svr}, RMSE: {rmse_svr}")
    
    # Collect overall scores
    overall_r2_scores_svr.append(r2_svr)
    overall_mse_scores_svr.append(mse_svr)
    overall_rmse_scores_svr.append(rmse_svr)
    
    # Combine predicted vs actual values into a single DataFrame
    results_df_svr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_svr, 'Activity': activity})
    combined_results_df_svr = pd.concat([combined_results_df_svr, results_df_svr])
    print(f"Predicted vs actual values combined for {activity}.")

    # Permutation importance
    perm_importance = permutation_importance(model_svr, X_test, y_test, n_repeats=10, random_state=42)
    
    # Store the feature importances
    feature_importances_svr[activity] = perm_importance.importances_mean
    
    # Plot permutation feature importances
    feature_importance_df_svr = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df_svr, palette='viridis')
    plt.title(f'Permutation Feature Importance for {activity} - SVM',fontsize=17, fontweight='bold')
    plt.xlabel('Importance',fontsize=14, fontweight='bold')
    plt.ylabel('Feature',fontsize=14, fontweight='bold')
    plt.show()
    print(f"Feature importance plot generated for {activity}.")

# Display only the first 5 values for all activities together
print("\nFirst 5 Predicted vs Actual values for all activities combined (SVM):\n")
print(combined_results_df_svr.head(5))

# Print the overall scores
print("\nOverall Scores for All Activities (SVM):")
print(f"Average R² Value: {sum(overall_r2_scores_svr) / len(overall_r2_scores_svr)}")
print(f"Average Mean Squared Error: {sum(overall_mse_scores_svr) / len(overall_mse_scores_svr)}")
print(f"Average Root Mean Squared Error: {sum(overall_rmse_scores_svr) / len(overall_rmse_scores_svr)}")

# Plotting the predicted vs actual values
plt.figure(figsize=(14, 10))
for activity in activities:
    plt.scatter(combined_results_df_svr[combined_results_df_svr['Activity'] == activity]['Actual'], combined_results_df_svr[combined_results_df_svr['Activity'] == activity]['Predicted'], label=activity, alpha=0.5)
plt.plot([combined_results_df_svr['Actual'].min(), combined_results_df_svr['Actual'].max()],
         [combined_results_df_svr['Actual'].min(), combined_results_df_svr['Actual'].max()], 'r--', lw=2)
plt.title('Predicted vs Actual Values (SVM)',fontsize=17, fontweight='bold')
plt.xlabel('Actual Values',fontsize=14, fontweight='bold')
plt.ylabel('Predicted Values',fontsize=14, fontweight='bold')
plt.legend(title='Activity',fontsize=14)
plt.show()
print("Predicted vs Actual plot generated.")

 #Use original column names for predictions
original_columns = [
    'Contacts Detail Age',
    'Contacts Detail Gender',
    'Attendance Detail Weekday',
    'Membership Level',
    'Contacts Detail Price Level'
]

# Function to get member details and predict activities for a given unique key using SVM
def get_member_details_and_predict_svr(unique_key):
    print(f"Fetching details for Unique Key: {unique_key}")
    
    # Search for the row with the given Unique Key
    member_data = data[data['Unique Key'] == unique_key]
    
    # Check if the member exists
    if member_data.empty:
        print(f"No data found for Unique Key: {unique_key}")
        return
    
    # Print the found data for debugging
    print("Selected member data:\n", member_data[['Unique Key', 'Frequency of Entry']])

    # Decode categorical columns to display
    member_data_display = member_data.copy()
    member_data_display['Contacts Detail Gender'] = label_encoders['Contacts Detail Gender'].inverse_transform(member_data['Contacts Detail Gender'])
    member_data_display['Membership Level'] = label_encoders['Membership Level'].inverse_transform(member_data['Membership Level'])
    member_data_display['Contacts Detail Price Level'] = label_encoders['Contacts Detail Price Level'].inverse_transform(member_data['Contacts Detail Price Level'])
    member_data_display['Entry Time Category'] = label_encoders['Entry Time Category'].inverse_transform(member_data['Entry Time Category'])
    member_data_display['Attendance Detail Weekday'] = label_encoders['Attendance Detail Weekday'].inverse_transform(member_data['Attendance Detail Weekday'])
    member_data_display['First Entry Month'] = label_encoders['First Entry Month'].inverse_transform(member_data['First Entry Month'])

    # Print the details
    print("\nMember Details:")
    print(f"Unique Key: {unique_key}")
    print(f"Gender: {member_data_display['Contacts Detail Gender'].values[0]}")
    print(f"Age: {member_data_display['Contacts Detail Age'].values[0]}")
    print(f"Membership Level: {member_data_display['Membership Level'].values[0]}")
    print(f"Price Level: {member_data_display['Contacts Detail Price Level'].values[0]}")
    print(f"Entry Time Category: {member_data_display['Entry Time Category'].values[0]}")
    print(f"Frequency of Entry: {member_data_display['Frequency of Entry'].values[0]}")
    print(f"Attendance Detail Weekday:{member_data_display['Attendance Detail Weekday'].values[0]}")
    print(f"First Entry Month: {member_data_display['First Entry Month'].values[0]}")
    
    # Predict activities using SVM
    input_data = member_data[original_columns]
    predictions_svr = {}
    for activity, model_svr in models_svr.items():
        predictions_svr[activity] = model_svr.predict(input_data)[0]
    
    print("\nPredicted Activities (SVM):")
    for activity, prediction in predictions_svr.items():
        print(f"{activity}: {prediction}")

# Example usage:
unique_id = int(input("Enter the Unique Key: "))
get_member_details_and_predict_svr(unique_id)