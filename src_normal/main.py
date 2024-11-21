import argparse

def main():
    parser = argparse.ArgumentParser(description="Sensor Testing and Validation Framework")
    parser.add_argument("--task", type=str, choices=["calibration", "vibration", "real_time", "data_collection", "nn_test"], required=True, help="Select the task to run")
    args = parser.parse_args()

    if args.task == "calibration":
        from calibration.calib_with_signal import CalibWithSignal
        calib = CalibWithSignal()
        calib.perform_calibration()
    elif args.task == "vibration":
        from calibration.vibration_test import VibrationTest
        test = VibrationTest()
        test.perform_vibration_test()
    elif args.task == "real_time":
        from real_time.ros_real_time_app import main as real_time_main
        real_time_main()
    elif args.task == "data_collection":
        from data_collection.reference_datacollection import FrankaDataCollection
        data_collector = FrankaDataCollection(data_dir="./data")
        data_collector.start_automation()
    elif args.task == "nn_test":
        from neural_network.neural_network_tester import NeuralNetworkTester
        tester = NeuralNetworkTester("tactile_sensor_model.h5", "collected_data.csv")
        tester.validate_model()

if __name__ == "__main__":
    main()
