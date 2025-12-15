import matplotlib.pyplot as plt

sizes = [40, 25, 20, 15]
labels = ['python', 'jave', 'golang', 'c']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Fruit Distribution")
plt.show()