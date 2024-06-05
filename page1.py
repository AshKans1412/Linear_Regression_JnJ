import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def Linear_Regression():
    # Title and description
    st.title('Linear Regression with Gradient Descent')
    st.write("This application demonstrates fitting a linear regression model using gradient descent, predicting height based on weight.")

    # Slider to select the number of data points
    num_points = st.slider('Select the number of data points:', min_value=50, max_value=1000, value=100, step=50)

    # Sliders to select mean and standard deviation of the noise
    noise_mean = st.slider('Select the mean of the noise:', min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
    noise_std = st.slider('Select the standard deviation of the noise:', min_value=0.0, max_value=20.0, value=10.0, step=0.5)


    # Generate synthetic data
    np.random.seed(42)
    x = np.random.normal(70, 15, num_points)

    mm = st.slider('Select the relationship between Height and Weight:', min_value=1.0, max_value=20.0, value=3.0, step=0.5)

    m_true = mm
    noise = np.random.normal(noise_mean, noise_std, num_points)
    y = m_true * x + noise


    # Initial plot of the data before fitting
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Initial Data Points')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    ax.legend()
    ax.set_title('Initial Distribution of Data')
    st.pyplot(fig)


    # Explanation of gradient descent with direct LaTeX rendering
    st.subheader('Gradient Descent Explanation for Simple Linear Regression (y = mx)')
    st.write("""
    The goal of gradient descent in the context of simple linear regression without an intercept is to minimize the loss function, which is typically the sum of squared residuals (errors). In this model, the prediction equation is simply \( y = mx \), where \( y \) is the predicted value based on the input \( x \), and \( m \) is the coefficient (slope) we are trying to learn.
    """)

    # Display the loss function equation using st.latex for better rendering
    st.latex(r"""
    L(m) = \sum_{i=1}^n (y_i - mx_i)^2
    """)

    st.write("""
    Where \( y_i \) and \( x_i \) are the observed values and input features, respectively. To find the value of \( m \) that minimizes \( L \), we calculate the derivative of \( L \) with respect to \( m \):
    """)

    st.latex(r"""
    \frac{\partial L}{\partial m} = -2 \sum x_i(y_i - mx_i)
    """)

    st.write("""
    In gradient descent, rather than solving this derivative equation analytically by setting it to zero and solving for \( m \), we use this gradient to iteratively adjust \( m \) towards the direction that reduces \( L \):
    """)

    st.latex(r"""
    m := m - \alpha \frac{\partial L}{\partial m}
    """)

    st.write("""
    This update is performed iteratively, where \( m \) is adjusted by a small step defined by the learning rate \( \alpha \). The process continues until the change in \( L \) between iterations is minimal, ideally converging to a local minimum where the gradient (derivative) is close to zero. The learning rate \( \alpha \) determines the size of each step and is a critical parameter that needs to be chosen carefully to ensure convergence.
    """)

    # Initialize parameters for gradient descent

    # Sliders for learning rate and number of iterations
    learning_rate = st.slider('Select the learning rate:', min_value=-0.00001, max_value=0.001, value=0.0001, step=0.00001, format='%.5f')
    iterations = st.slider('Select the number of iterations:', min_value=100, max_value=10000, value=1000, step=100)
    initial_m = st.slider('Select the initial value of m:', min_value=-100.0, max_value=100.0, value=0.1, step=0.1)

    m = initial_m
    m_values = []

    # Perform gradient descent
    for i in range(iterations):
        y_pred = m * x
        gradients = -2 * np.dot(x, (y - y_pred)) / len(x)
        m -= learning_rate * gradients
        m_values.append(m)

    # Plotting the loss function parabola
    m_range = np.linspace(0, 6, 400)
    loss = lambda m: np.sum((y - m * x) ** 2)
    loss_values = [loss(mi) for mi in m_range]

    fig, ax = plt.subplots()
    ax.plot(m_range, loss_values, label='Loss Function')
    ax.scatter(m_values[::int(len(m_values)/10)], [loss(m) for m in m_values[::int(len(m_values)/10)]], color='red', zorder=5, label='m at intervals')
    ax.set_title('Loss Function Over Iterations')
    ax.set_xlabel('m value')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Plotting the final model fitting
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    ax.plot(x, m * x, color='red', label=f'Fitted line: y = {m:.2f}x')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    ax.legend()
    ax.set_title('Fitted Line After Gradient Descent')
    st.pyplot(fig)

    # Display parameters and iterations
    st.write(f"Final estimated slope (m): {m:.2f}")
    st.line_chart(m_values)