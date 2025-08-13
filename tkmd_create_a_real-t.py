import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class TKMD_RealTimeModelIntegrator:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Machine Learning Model Integrator")
        self.root.geometry("800x600")

        # Create frames
        self.model_frame = tk.Frame(self.root, bg="gray")
        self.model_frame.pack(fill="both", expand=True)

        self.data_frame = tk.Frame(self.root, bg="gray")
        self.data_frame.pack(fill="both", expand=True)

        # Create data frame widgets
        self.data_label = tk.Label(self.data_frame, text="Select Data File:")
        self.data_label.pack(fill="x")

        self.data_entry = tk.Entry(self.data_frame, width=50)
        self.data_entry.pack(fill="x")

        self.browse_button = tk.Button(self.data_frame, text="Browse", command=self.browse_data_file)
        self.browse_button.pack(fill="x")

        # Create model frame widgets
        self.model_label = tk.Label(self.model_frame, text="Select Model File:")
        self.model_label.pack(fill="x")

        self.model_entry = tk.Entry(self.model_frame, width=50)
        self.model_entry.pack(fill="x")

        self.browse_model_button = tk.Button(self.model_frame, text="Browse", command=self.browse_model_file)
        self.browse_model_button.pack(fill="x")

        self.train_button = tk.Button(self.model_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(fill="x")

        self.integrate_button = tk.Button(self.model_frame, text="Integrate Model", command=self.integrate_model)
        self.integrate_button.pack(fill="x")

        self.status_label = tk.Label(self.model_frame, text="Status:")
        self.status_label.pack(fill="x")

        self.status_text = tk.Text(self.model_frame, height=10, width=50)
        self.status_text.pack(fill="both", expand=True)

    def browse_data_file(self):
        file_path = filedialog.askopenfilename()
        self.data_entry.delete(0, tk.END)
        self.data_entry.insert(0, file_path)

    def browse_model_file(self):
        file_path = filedialog.askopenfilename()
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, file_path)

    def train_model(self):
        data_path = self.data_entry.get()
        if not data_path:
            messagebox.showerror("Error", "Please select a data file")
            return

        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"Model trained with accuracy: {accuracy:.2f}\n")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

    def integrate_model(self):
        model_path = self.model_entry.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found")
            return

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "Model integrated successfully\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TKMD_RealTimeModelIntegrator(root)
    root.mainloop()