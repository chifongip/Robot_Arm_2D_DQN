import subprocess

script1_path = "model_FCNN.py"
script2_path = "model_CNN1.py"
script3_path = "model_CNN2.py"
script4_path = "model_LSTM.py"

subprocess.run(["python", script1_path])

subprocess.run(["python", script2_path])

subprocess.run(["python", script3_path])

subprocess.run(["python", script4_path])
