import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [10, 20, 25, 30, 40]
y2 = [1, 2, 3, 5, 7]

# Create the first plot
fig, ax1 = plt.subplots()

# Plot the first line
ax1.plot(x, y1, 'g-', label='Line 1')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y1 axis', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot the second line
ax2.plot(x, y2, 'b-', label='Line 2')
ax2.set_ylabel('Y2 axis', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Show the plot
plt.title("Plot with Two Different Y Axes")
plt.show()