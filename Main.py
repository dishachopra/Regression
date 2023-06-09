import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import regdata as rd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

degree= st.sidebar.slider('Polynomial Degree', 1, 20, 1)

degree1 = st.sidebar.slider('sin(x) coefficient', 0, 200, 15)

degree2 = st.sidebar.slider('sin(x) and cos(x) coefficient', 0, 200, 5)


x = np.linspace(-2 * np.pi, 2 * np.pi, 20)
np.random.seed(42)
noise = np.random.normal(0, 2, len(x))
y_org= np.sin(x)+4*x**2+5*x
y = y_org + noise

X = x.reshape(-1, 1)
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 200)
X_pred = x_pred.reshape(-1, 1)
X_pred_poly = poly_features.transform(X_pred)
y_actual = np.sin(X_pred)+ 4*X_pred**2 +5*X_pred

y_pred = model.predict(X_pred_poly)
y_train_pred=model.predict(X_poly)

coefficients = model.coef_
intercept = model.intercept_

# Generate the regression equation string
regression_equation = "y = {:.2f}".format(intercept)

for i in range(1, degree + 1):
    regression_equation += " + {:.2f} * x^{}".format(coefficients[i], i)

# Print the regression equation
checkbox_value = st.checkbox("Show regression equation")
if checkbox_value:
    st.write("Regression equation:", regression_equation, font_size=29)

checkbox_value = st.checkbox("Decomposition of the regression equation")
if checkbox_value:
    plt.figure(figsize=(10, 4 ))
    plt.plot(x, intercept * np.ones_like(x))
    plt.xlabel("x")
    plt.ylabel(r"$\theta_0$")
    plt.title(r"$\theta_0$ vs x")
    plt.legend()
    st.pyplot(plt)

# Plotting thetai * x vs x
    plt.figure(figsize=(10, 4 * (degree + 1)))
    for i in range(1, degree + 1):
        plt.subplot(degree + 1, 1, i + 1)
        plt.plot(x, coefficients[i] * x ** i)
        plt.xlabel("x")
        plt.ylabel(r"$\theta_{}$ * $x^{}$".format(i, i))
        plt.title(r"$\theta_{}$ * $x^{}$ vs x".format(i, i))
        plt.legend()

# Adjust spacing between subplots
    plt.tight_layout()

# Display the plot
    st.pyplot(plt)
# Plotting the actual sin(x)+ 4x²+ 5x function and the predicted values
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='g', label='Actual')
plt.plot(x_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x) + 4x² + 5x')
plt.title('Polynomial Regression for sin(x) + 4x² + 5x + noise')
plt.legend()

# Display the second plot
st.pyplot(plt)

mse = mean_squared_error(y, y_train_pred)
rmse = math.sqrt(mse)
rmse_rounded = round(rmse, 2)
st.write("RMSE train:")
st.write(rmse_rounded)

mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
rmse_rounded = round(rmse, 2)
st.write("RMSE test:")
st.write(rmse_rounded)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

y_pred = model.predict(X_pred)
y_train_pred = model.predict(X)

# Plotting the actual sin(x) + 4x² + 5x function and the predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X, y , color='g', label='Actual')
plt.plot(X_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x) + 4x² + 5x')
plt.title('Random Forest Regression for sin(x) + 4x² + 5x + noise')
plt.legend()

# Display the plot using Streamlit
st.pyplot(plt)

# Calculate RMSE for train set
mse_train = mean_squared_error(y, y_train_pred)
rmse_train = math.sqrt(mse_train)

# Calculate RMSE for test set
y_actual = np.sin(X_pred) + 4 * X_pred ** 2 + 5 * X_pred
mse_test = mean_squared_error(y_actual, y_pred)
rmse_test = math.sqrt(mse_test)

# Display RMSE values using Streamlit
st.write("RMSE (Train):", rmse_train)
st.write("RMSE (Test):", rmse_test)





X_custom = np.empty((len(x), 0))
X_custom = np.column_stack((X_custom, x))
for i in np.arange(1, degree1 + 1,1/16):
    X_custom = np.column_stack((X_custom, np.sin((i *np.pi)+ x)))
    



model = LinearRegression()
model.fit(X_custom, y)




coefficients = model.coef_
intercept = model.intercept_

regression_equation = "y = {:.2f}".format(intercept)
if coefficients[0] != 0:
    regression_equation += " + {:.2f}x".format(coefficients[0])

for i, coefficient in enumerate(coefficients[1:], start=1):
    regression_equation += " + {:.2f} * sin({}x)".format(coefficient, i)

checkbox_value = st.checkbox("Show Regression equation")
if checkbox_value:
    st.write("Regression equation:", regression_equation)

x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 20)

X_pred_custom = np.empty((len(x_pred), 0))
X_pred_custom = np.column_stack((X_pred_custom, x_pred))
for i in np.arange(1, degree1 + 1, 1/16):
    X_pred_custom = np.column_stack((X_pred_custom, np.sin((i*np.pi)+ x_pred)))
    

y_pred = model.predict(X_pred_custom)
y_train_pred=model.predict(X_custom)
y_actual = np.sin(x_pred) + 4 * x_pred ** 2 + 5 * x_pred


# Plot the actual sin(x) function and the predicted values
plt.figure()
# plt.plot(x, y, color='r', label='Actual')
# plt.scatter(x_pred , y_pred, color='r', label='Actual')
# plt.plot(x_pred, y_pred, color='b', label='Predicted')
plt.scatter(x, y, color='g', label='Actual')
plt.plot(x_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x)+ 4x²+ 5x')
plt.title('Polynomial Regression with Custom Features for sin(x)')
plt.legend()
plt.show()
st.pyplot(plt)

# Calculate the MSE and RMSE

mse_train = mean_squared_error(y, y_train_pred)
rmse_train = math.sqrt(mse_train)
rmse_train_rounded = round(rmse_train, 2)
st.write("RMSE Train:")
st.write(rmse_train_rounded)


mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
rmse_rounded = round(rmse, 2)
st.write("RMSE Test:")
st.write(rmse_rounded)


X_custom = np.empty((len(x), 0))
X_custom = np.column_stack((X_custom, x))
for i in range(1, degree2 + 1):
    X_custom = np.column_stack((X_custom, np.sin(i * x), np.cos((i * x))))

# min_max_scalar  = MinMaxScaler()


model = MLPRegressor(hidden_layer_sizes=(100,100,2), random_state=42,max_iter=2000, learning_rate_init=0.01)
model.fit(X_custom, y)


x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 20)

X_pred_custom = np.empty((len(x_pred), 0))
X_pred_custom = np.column_stack((X_pred_custom, x_pred))
for i in range(1, degree2 + 1):
    X_pred_custom = np.column_stack((X_pred_custom, np.sin(i* x_pred), np.cos(i*x_pred) ))

y_pred = model.predict(X_pred_custom)
y_train_pred=model.predict(X_custom)
y_actual = np.sin(x_pred) + 4 * x_pred ** 2 + 5 * x_pred





plt.figure()
# plt.plot(x, y, color='r', label='Actual')
# plt.scatter(x_pred , y_pred, color='r', label='Actual')
# plt.plot(x_pred, y_pred, color='b', label='Predicted')
plt.scatter(x, y, color='g', label='Actual')
plt.plot(x_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x)+ 4x²+ 5x')
plt.title('MLP with Custom Features for sin(x) and cos(x)')
plt.legend()
plt.show()
st.pyplot(plt)

# Calculate the MSE and RMSE

mse_train = mean_squared_error(y, y_train_pred)
rmse_train = math.sqrt(mse_train)
rmse_train_rounded = round(rmse_train, 2)
st.write("RMSE Train:")
st.write(rmse_train_rounded)


mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
rmse_rounded = round(rmse, 2)
st.write("RMSE Test:")
st.write(rmse_rounded)

X_custom = np.empty((len(x), 0))
X_custom = np.column_stack((X_custom, x))
for i in range(1, degree2 + 1):
    X_custom = np.column_stack((X_custom, np.sin(i * x), np.sin(np.pi/4- i*x)))


model = LinearRegression()
model.fit(X_custom, y)



# Get the coefficients of the linear regression model
coefficients = model.coef_
intercept = model.intercept_

regression_equation = "y = {:.2f}".format(intercept)
if coefficients[0] != 0:
    regression_equation += " + {:.2f}x".format(coefficients[0])

for i in range(1, degree2 + 1):
    regression_equation += " + {:.2f} * sin({}x) + {:.2f} * cos({}x)".format(coefficients[2 * i - 1], i, coefficients[2 * i], i)

checkbox_value = st.checkbox("Show Regression Equation")
if checkbox_value:
    st.write("Regression equation:", regression_equation)

x_pred = np.linspace(-2 * np.pi, 2 * np.pi, 20)

X_pred_custom = np.empty((len(x_pred), 0))
X_pred_custom = np.column_stack((X_pred_custom, x_pred))
for i in range(1, degree2 + 1):
    X_pred_custom = np.column_stack((X_pred_custom, np.sin(i* x_pred), np.sin(np.pi/4-i*x_pred) ))

y_pred = model.predict(X_pred_custom)
y_train_pred=model.predict(X_custom)
y_actual = np.sin(x_pred) + 4 * x_pred ** 2 + 5 * x_pred

checkbox_value2 = st.checkbox("Decomposition of regression equation")
if checkbox_value2:

    plt.figure(figsize=(10, 4 ))
    plt.plot(x, intercept * np.ones_like(x))
    plt.xlabel("x")
    plt.ylabel(r"$\theta_0$")
    plt.title(r"$\theta_0$ vs x")
    plt.legend()
    st.pyplot(plt)

# Plotting thetai * x vs x
    plt.figure(figsize=(10, 4 * (degree2 + 1)))
    for i in range(1, degree2 + 1):
        plt.subplot(degree2 + 1, 1, i + 1)
        plt.plot(x, coefficients[i] * x ** i)
        plt.xlabel("x")
        plt.ylabel(r"$\theta_{}$ * $x^{}$".format(i, i))
        plt.title(r"$\theta_{}$ * $x^{}$ vs x".format(i, i))
        plt.legend()

# Adjust spacing between subplots
    plt.tight_layout()

# Display the plot
    st.pyplot(plt)



plt.figure()
# plt.plot(x, y, color='r', label='Actual')
# plt.scatter(x_pred , y_pred, color='r', label='Actual')
# plt.plot(x_pred, y_pred, color='b', label='Predicted')
plt.scatter(x, y, color='g', label='Actual')
plt.plot(x_pred, y_pred, color='r', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(x)+ 4x²+ 5x')
plt.title('Polynomial Regression with Custom Features for sin(x) and cos(x)')
plt.legend()
plt.show()
st.pyplot(plt)

# Calculate the MSE and RMSE

mse_train = mean_squared_error(y, y_train_pred)
rmse_train = math.sqrt(mse_train)
rmse_train_rounded = round(rmse_train, 2)
st.write("RMSE Train:")
st.write(rmse_train_rounded)


mse = mean_squared_error(y_actual, y_pred)
rmse = math.sqrt(mse)
rmse_rounded = round(rmse, 2)
st.write("RMSE Test:")
st.write(rmse_rounded)



# Define a dictionary of available datasets
datasets = {
    'Jump1D': rd.Jump1D,
    'Heinonen4': rd.Heinonen4,
    'DellaGattaGene': rd.DellaGattaGene,
    'MotorcycleHelmet': rd.MotorcycleHelmet,
    'Olympic': rd.Olympic,
    'SineNoisy': rd.SineNoisy,
    'Smooth1D': rd.Smooth1D,
}

# Create a sidebar selectbox to choose the dataset
dataset_name = st.sidebar.selectbox('Select Dataset', list(datasets.keys()))

# Get the dataset based on the selected name
dataset = datasets[dataset_name]()

# Get the data for training and testing
X, y_train, X_test = dataset.get_data()

# Set the degree for polynomial regression
degree3 = st.sidebar.slider('sin(x) and cos(x)', 0, 30, 5)

# Create the custom features matrix for training set
X_train_custom = np.empty((len(X), 0))
X_train_custom = np.column_stack((X_train_custom, X))
for i in range(1, degree3 + 1):
    X_train_custom = np.column_stack((X_train_custom, np.sin(i * X), np.cos(i * X)))

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train_custom, y_train)

# Create the custom features matrix for test set
X_test_custom = np.empty((len(X_test), 0))
X_test_custom = np.column_stack((X_test_custom, X_test))
for i in range(1, degree3 + 1):
    X_test_custom = np.column_stack((X_test_custom, np.sin(i * X_test), np.cos(i * X_test)))

# Predict on the training set
y_train_pred = model.predict(X_train_custom)

# Predict on the test set
y_test_pred = model.predict(X_test_custom)

# Plot the actual and predicted values
plt.figure()
plt.scatter(X, y_train, color='g', label='Actual')
plt.plot(X, y_train_pred, color='r', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Custom Features')
plt.legend()
plt.show()
st.pyplot(plt)

# Calculate the RMSE for the training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
st.write("RMSE Train:", round(rmse_train, 2))

# Calculate the RMSE for the test set
# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
# st.write("RMSE Test:", round(rmse_test, 2))
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_train)
y_train_pred = model.predict(X)


plt.figure()
plt.scatter(X, y_train, color='g', label='Actual')
plt.plot(X, y_train_pred, color='r', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Random Forest')
plt.legend()
plt.show()
st.pyplot(plt)

# Calculate the RMSE for the training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
st.write("RMSE Train (rf):", round(rmse_train, 2))


import GPy

N = 100

X = np.linspace(-1, 1, N).reshape(-1, 1)

# @st.cache(show_spinner=False)
def generate_gp_sample(variance, lengthscale, seed):
    np.random.seed(seed)
    kernel = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    covariance = kernel.K(X, X)
    y = np.random.multivariate_normal(np.zeros(N), covariance)
    y_noisy = y + np.random.normal(0, np.sqrt(variance) / 10, size=N)
    return y, y_noisy


st.sidebar.title("Gaussian Process Regression")


y_true, y_noisy = generate_gp_sample(0.1, 0.1, 0)

degree4 = st.sidebar.slider('sin(x) and cos(x) for dataset', 0, 20, 5)

X_train_custom = np.empty((len(X), 0))
X_train_custom = np.column_stack((X_train_custom, X))
for i in range(1, degree4 + 1):
    X_train_custom = np.column_stack((X_train_custom, np.sin(i * X), np.cos(i * X)))
y_train = y_noisy[:len(X_train_custom)]

model = LinearRegression()
model.fit(X_train_custom, y_train)
y_pred = model.predict(X_train_custom)

fig2, ax2 = plt.subplots()
ax2.plot(X, y_true, label="True function")
ax2.scatter(X, y_noisy, label="Data")
ax2.plot(X, y_pred, color='r', label='Polynomial Regression')
ax2.legend()
st.pyplot(fig2)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
st.write("RMSE Train:", round(rmse_train, 2))

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_noisy)
y_pred = model.predict(X)

fig2, ax2 = plt.subplots()
ax2.plot(X, y_true, label="True function")
ax2.scatter(X, y_noisy, label="Data")
ax2.plot(X, y_pred, color='r', label='Random Forest')
ax2.legend()
st.pyplot(fig2)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
st.write("RMSE Train(rf):", round(rmse_train, 2))


import pickle

# Load in the data
with open("mauna_loa", "rb") as fid:
    data = pickle.load(fid)

# Training data (X = input, Y = observation)
#take only 200 columns of data

X, Y = data['X'][:, :50], data['Y'][:, :500]

# Test data (Xtest = input, Ytest = observations)
Xtest, Ytest = data['Xtest'][:, :100], data['Ytest'][:, :100]


# Apply polynomial regression   
degree5 = st.sidebar.slider('Polynomial Degree', 1, 200, 5)      
poly_features = PolynomialFeatures(degree=degree5)
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, Y)        

# Generate predictions for the training and test sets 

#
X_pred = np.linspace(1960, 2010, 200).reshape(-1, 1)
X_pred_poly = poly_features.transform(X_pred)
Y_pred = model.predict(X_pred_poly)

# Plot the actual data and the predicted values
plt.figure(figsize=(14, 8))
plt.plot(X, Y, "b.")
plt.plot(X_pred, Y_pred, "g-")
plt.legend(labels=["training data", "test data", "predictions"])
plt.xlabel("year")
plt.ylabel("CO$_2$ (PPM)")
plt.title("Polynomial Regression")
st.pyplot(plt)
#Get the RMSE for the training set
Y_train_pred = model.predict(X_poly)
rmse_train = np.sqrt(mean_squared_error(Y, Y_train_pred))
st.write("RMSE Train(PR):", round(rmse_train, 2))


#Apply regression with custom features of sinx and cosx
degree6 = st.sidebar.slider('sin(x) and cos(x) for Dataset', 0, 200, 5)
X_custom = np.empty((len(X), 0))
X_custom = np.column_stack((X_custom, X))
for i in range(-10, degree6 + 1):
    X_custom = np.column_stack((X_custom, np.sin((2**(i*np.pi)) * X), np.cos((2**(i*np.pi)) * X)))
model = LinearRegression()
model.fit(X_custom, Y)
X_pred_custom = np.empty((len(X_pred), 0))
X_pred_custom = np.column_stack((X_pred_custom, X_pred))
for i in range(-10, degree6 + 1):
    X_pred_custom = np.column_stack((X_pred_custom, np.sin((2**(i*np.pi)) * X_pred), np.cos((2**(i*np.pi)) * X_pred)))
Y_pred = model.predict(X_pred_custom)
plt.figure(figsize=(14, 8))
plt.plot(X, Y, "b.")
plt.plot(X_pred, Y_pred, "g-")
plt.legend(labels=["training data", "test data", "predictions"])
plt.xlabel("year")
plt.ylabel("CO$_2$ (PPM)")
plt.title("Custom features of sinx and cosx")
st.pyplot(plt)

#Get the RMSE for the training set
Y_train_pred = model.predict(X_custom)
rmse_train = np.sqrt(mean_squared_error(Y, Y_train_pred))
st.write("RMSE Train:", round(rmse_train, 2))


# Apply random forest regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)
Y_pred = model.predict(X_pred)
plt.figure(figsize=(14, 8))
plt.plot(X, Y, "b.")
plt.plot(X_pred, Y_pred, "g-")
plt.legend(labels=["training data", "test data", "predictions"])
plt.xlabel("year")
plt.ylabel("CO$_2$ (PPM)")
plt.title("Random Forest")
st.pyplot(plt)
#Get the RMSE for the training set
Y_train_pred = model.predict(X)
rmse_train = np.sqrt(mean_squared_error(Y, Y_train_pred))
st.write("RMSE Train(rf):", round(rmse_train, 2))











