import tkinter as tk  # create the main application window

# Create window
root = tk.Tk()
root.title("Simple Tkinter App")
root.geometry("200x100")  # use lowercase x

# Function to print "Hello, World!" in the console
def say_hello():
    print("Hello, World!")

# Create a button that triggers the say_hello function
hello_button = tk.Button(root, text="Click Me", command=say_hello)
hello_button.pack(pady=20)  # Pack the button into the window

# Start the Tkinter event loop
root.mainloop()
