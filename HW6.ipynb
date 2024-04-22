import pandas as pd

file_path = '/Users/linzeru/Downloads/employee.csv'  # Adjust path as needed
employee_data = pd.read_csv(file_path)

# Define features and target variable
X = employee_data.drop(columns=['salary', 'id', 'timestamp'])
y = employee_data['salary']

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Create preprocessors for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply preprocessing
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# Initialize Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_prepared, y_train)

# Make predictions on test data
y_pred_linear = linear_model.predict(X_test_prepared)

# Calculate MAE and MSE for Linear Regression
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)

# Initialize and train Ridge and Lasso regression models
ridge_model = Ridge()
lasso_model = Lasso()

# Train Ridge and Lasso models
ridge_model.fit(X_train_prepared, y_train)
lasso_model.fit(X_train_prepared, y_train)

# Make predictions with Ridge and Lasso
y_pred_ridge = ridge_model.predict(X_test_prepared)
y_pred_lasso = lasso_model.predict(X_test_prepared)

# Calculate MAE and MSE for Ridge and Lasso
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print the results
print("Linear Regression MAE:", mae_linear)
print("Linear Regression MSE:", mse_linear)
print("Ridge Regression MAE:", mae_ridge)
print("Ridge Regression MSE:", mse_ridge)
print("Lasso Regression MAE:", mae_lasso)
print("Lasso Regression MSE:", mse_lasso)


# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Splitting data into features and target variable
X = employee_data.drop('salary', axis=1)
y = employee_data['salary']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pipeline for Linear Regression
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Pipeline for Ridge Regression
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# Pipeline for Lasso Regression
lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1.0))
])

# Fit and predict with Linear Regression
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Fit and predict with Ridge Regression
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Fit and predict with Lasso Regression
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate models
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print evaluation metrics
print("Linear Regression MAE:", mae_lr)
print("Linear Regression MSE:", mse_lr)
print("Ridge Regression MAE:", mae_ridge)
print("Ridge Regression MSE:", mse_ridge)
print("Lasso Regression MAE:", mae_lasso)
print("Lasso Regression MSE:", mse_lasso)