import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_with_intercept():
    st.title('Linear Regression with Intercept (y = mx + b)')
    
    st.write("""
    ### Gradient Descent Explanation
    For the linear regression model with an intercept, the model is defined as \( y = mx + b \). Here, \( m \) is the slope of the line, and \( b \) is the y-intercept. We aim to find the values of \( m \) and \( b \) that minimize the sum of squared residuals, which is the difference between the observed values and the values predicted by our model.

    The loss function \( L \) is defined as:
    """)
    st.latex(r"L = \sum_{i=1}^n (y_i - (mx_i + b))^2")

    st.write("""
    To minimize \( L \), we perform gradient descent, updating \( m \) and \( b \) iteratively using partial derivatives of \( L \) with respect to \( m \) and \( b \):
    """)
    st.latex(r"\frac{\partial L}{\partial m} = -2 \sum_{i=1}^n x_i(y_i - (mx_i + b))")
    st.latex(r"\frac{\partial L}{\partial b} = -2 \sum_{i=1}^n (y_i - (mx_i + b))")

    st.write("""
    In each iteration of gradient descent, \( m \) and \( b \) are updated as follows:
    """)
    st.latex(r"m := m - \alpha \frac{\partial L}{\partial m}")
    st.latex(r"b := b - \alpha \frac{\partial L}{\partial b}")

    st.write("Where \( \alpha \) is the learning rate, a parameter that determines the step size during each iteration.")

    # UI for parameter selection
    num_points = st.slider('Select the number of data points:', 50, 1000, 100, 50, key='num_points2')
    noise_mean = st.slider('Mean of the noise:', -20.0, 20.0, 0.0, 0.5, key='noise_mean2')
    noise_std = st.slider('Standard deviation of the noise:', 0.0, 20.0, 10.0, 0.5, key='noise_std2')
    intercept = st.slider('Intecept of the eqn:', min_value=0.0, max_value=100.0, value=20.0, step=0.5)
    slope = st.slider('Intecept of the eqn:', min_value=0.0, max_value=20.0, value=3.0, step=0.5)

    # Generate data
    x = np.random.normal(70, 15, num_points)
    y = (slope * x) + intercept + np.random.normal(noise_mean, noise_std, num_points)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Initial Data Points')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    st.pyplot(fig)

    initial_m = st.slider('Initial value of m:', 0.0, 10.0, 0.1, 0.1, key='initial_m2')
    initial_b = st.slider('Initial value of b:', -10.0, 10.0, 0.0, 0.1, key='initial_b2')
    learning_rate = st.slider('Learning rate:', 0.00001, 0.01, 0.0001, 0.00001, format='%.5f', key='learning_rate2')
    iterations = st.slider('Number of iterations:', 100, 10000, 1000, 100, key='iterations2')
    # Gradient descent
    m, b = initial_m, initial_b
    m_values, b_values = [], []
    for _ in range(iterations):
        y_pred = m * x + b
        gradient_m = -2 * np.dot(x, (y - y_pred)) / len(x)
        gradient_b = -2 * np.sum(y - y_pred) / len(x)
        m -= learning_rate * gradient_m
        b -= learning_rate * gradient_b
        m_values.append(m)
        b_values.append(b)

    # Display the final model fitting
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    ax.plot(x, m * x + b, color='red', label=f'Fitted line: y = {m:.2f}x + {b:.2f}')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    ax.legend()
    ax.set_title('Fitted Line After Gradient Descent')
    st.pyplot(fig)

    # Loss function calculation for m and b ranges
    m_range = np.linspace(min(m_values) - 1, max(m_values) + 1, 100)
    b_range = np.linspace(min(b_values) - 1, max(b_values) + 1, 100)
    m_loss = [np.sum((y - mi * x + b) ** 2) for mi in m_range]
    b_loss = [np.sum((y - m * x + bi) ** 2) for bi in b_range]
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for m
    axs[0].plot(m_range, m_loss, 'r-', label='Loss as a function of m')
    axs[0].set_title('Loss vs. Slope (m)')
    axs[0].set_xlabel('Slope (m)')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot for b
    axs[1].plot(b_range, b_loss, 'b-', label='Loss as a function of b')
    axs[1].set_title('Loss vs. Intercept (b)')
    axs[1].set_xlabel('Intercept (b)')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    st.pyplot(fig)

    # Final model plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    ax.plot(x, m * x + b, color='red', label=f'Final line: y = {m:.2f}x + {b:.2f}')
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Height (cm)')
    ax.legend()
    ax.set_title('Final Fitted Line After Gradient Descent')
    st.pyplot(fig)




