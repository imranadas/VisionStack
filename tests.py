import tensorflow as tf
import torch
import onnx
import onnxruntime
import os
import time
import csv
from ultralytics import YOLO

class GPUChecker:
    def __init__(self, framework):
        self.framework = framework

    def check_gpu(self):
        if self.framework == "tensorflow":
            self._check_tensorflow_gpu()
        elif self.framework == "pytorch":
            self._check_pytorch_gpu()
        elif self.framework == "onnx":
            self._check_onnx_gpu()
        else:
            print("Unknown framework.")

    def _check_tensorflow_gpu(self):
        if tf.test.is_gpu_available():
            print("TensorFlow GPU support is available.")
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                print(f"GPU Name: {device.name}")
        else:
            print("TensorFlow GPU not found. Using CPU.")

    def _check_pytorch_gpu(self):
        if torch.cuda.is_available():
            print("PyTorch GPU support is available.")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU Name: {gpu_name}")
        else:
            print("PyTorch GPU not found. Using CPU.")

    def _check_onnx_gpu(self):
        device = onnxruntime.get_device()
        print(device)
        if device == 'GPU':
            print("ONNX GPU support is available.")
            gpu_name = onnxruntime.get_available_providers()
            print(f"Available Execution Providers: {gpu_name}")
        else:
            print("ONNX GPU not found. Using CPU.")

class ModelBenchmark:
    def __init__(self, models, dataset_folder, total_times_csv_file_path):
        self.models = models
        self.dataset_folder = dataset_folder
        self.total_times_csv_file_path = total_times_csv_file_path

    def run_benchmark(self):
        with open(self.total_times_csv_file_path, mode='w', newline='') as total_times_csv_file:
            total_times_csv_writer = csv.writer(total_times_csv_file)
            total_times_csv_writer.writerow(['Model', 'Total Time (seconds)'])
            for model_name in self.models:
                model = YOLO(model_name)
                total_time = 0
                individual_times_csv_file_path = f"individual_times_{model_name}.csv"
                with open(individual_times_csv_file_path, mode='w', newline='') as individual_times_csv_file:
                    individual_times_csv_writer = csv.writer(individual_times_csv_file)
                    individual_times_csv_writer.writerow(['Image', 'Time (seconds)'])
                    for root, dirs, files in os.walk(self.dataset_folder):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(root, file)
                                start_time = time.time()
                                results = model.predict(img_path, save=False)
                                end_time = time.time()
                                total_time += (end_time - start_time)
                                individual_times_csv_writer.writerow([file, end_time - start_time])
                total_times_csv_writer.writerow([model_name, total_time])
        print(f"Total times have been exported to {self.total_times_csv_file_path}")
        print(f"Individual times for each model have been exported to CSV files")

if __name__ == "__main__":
    frameworks = ["tensorflow", "pytorch", "onnx"]
    for framework in frameworks:
        print(f"Checking {framework.capitalize()} GPU support:")
        gpu_checker = GPUChecker(framework)
        gpu_checker.check_gpu()

    models = ["Freeze_Best.onnx", "Freeze_Best.pt", "FullyRetrain_Best.onnx", "FullyRetrain_Best.pt"]
    dataset_folder = "Dataset_RAW"
    total_times_csv_file_path = "total_times.csv"
    benchmark = ModelBenchmark(models, dataset_folder, total_times_csv_file_path)
    benchmark.run_benchmark()
