import customtkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

def create_preprocessing_page(frame):
    # Example data
    raw_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    processed_data = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})

    # Display raw data
    raw_data_label = customtkinter.CTkLabel(frame, text="Raw Data", font=("Helvetica", 14))
    raw_data_label.pack(pady=10)
    raw_data_text = customtkinter.CTkTextbox(frame, width=300, height=100)
    raw_data_text.insert("1.0", raw_data.to_string())
    raw_data_text.pack(pady=10)

    # Display processed data
    processed_data_label = customtkinter.CTkLabel(frame, text="Processed Data", font=("Helvetica", 14))
    processed_data_label.pack(pady=10)
    processed_data_text = customtkinter.CTkTextbox(frame, width=300, height=100)
    processed_data_text.insert("1.0", processed_data.to_string())
    processed_data_text.pack(pady=10)

    # Example plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 30])

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
