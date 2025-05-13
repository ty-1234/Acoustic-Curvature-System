## Data and Filenaming Convention

The dataset for this project undergoes a two-stage preprocessing pipeline.

### Stage 1: Raw Data Synchronization (`csv_sync.py`)

Raw data is collected from two primary sources:
1.  **Audio FFT Data**: Contains Fast Fourier Transform features extracted from acoustic signals. Filenames are expected in the format `raw_audio_<CURVATURE_STR>[TEST_ID].csv` (e.g., `raw_audio_0_01[test 1].csv`).
2.  **Robot Positional Data**: Contains robot arm position, clamp status ("open"/"close"), and sensor section/position (`Position_cm`). Filenames are expected in the format `raw_robot_<CURVATURE_STR>[TEST_ID].csv` (e.g., `raw_robot_0_01[test 1].csv`).

The `<CURVATURE_STR>` component in these raw filenames represents the nominal target curvature value applied during that specific experiment, with underscores replacing decimal points (e.g., `0.01` becomes `0_01`, `0.007142` becomes `0_007142`).

The `[TEST_ID]` component (e.g., `[test 1]`, `[test 2]`, `[test 3]`, `[test 4]`) identifies the specific **Test Object** used for that data collection run. We utilize 5 distinct physical test objects, each designed to impart specific curvature profiles.

The script `curvature_sensor_moveit/src_normal/curvature_sensor/scripts/csv_sync.py` is responsible for:
* Identifying pairs of `raw_audio_` and `raw_robot_` files based on matching `<CURVATURE_STR>` and `[TEST_ID]`.
* Synchronizing these files based on timestamps.
* Identifying "active" periods where the robot clamp is engaged ("close" to "open" events).
* Populating the following key columns in the output:
    * `FFT_200Hz` to `FFT_2000Hz`: From the audio data.
    * `Timestamp`: Synchronized timestamp.
    * `Curvature`: Set to the nominal curvature value (derived from the `<CURVATURE_STR>` in the filename) for rows where `Curvature_Active` is 1, and 0.0 otherwise.
    * `Curvature_Active`: Set to `1` for data points recorded during an active robot clamp interval, `0` otherwise.
    * `Position_cm`: The sensor position (0cm to 5cm) as recorded by the robot during active intervals; NaN otherwise.

The output files from this stage are saved in the `csv_data/merged/` directory with the naming convention:
`merged_<CURVATURE_STR>[TEST_ID].csv`

Example:
* `merged_0_01[test 1].csv` - Data for curvature 0.01 m<sup>-1</sup> using Test Object 1.
* `merged_0_05[test 4].csv` - Data for curvature 0.05 m<sup>-1</sup> using Test Object 4.

We have 5 distinct curvature profiles (0.005, 0.007142, 0.01, 0.01818, 0.05 m<sup>-1</sup>) and 4 Test Objects, resulting in 20 `merged_*.csv` files.

### Stage 2: Noise Removal & Segment Cleaning (`noise_remover.py`)

The `merged_*.csv` files are then copied to this folder to be processed by `noise_remover.py` (containing the `process_data_for_discrete_extraction` function). This script:
* Identifies all distinct blocks where `Curvature_Active == 1`.
* For each active block:
    * Deletes approximately 1 second worth of rows immediately *before* the start of that block.
    * Deletes the last ~1 second worth of rows *from within the active block itself*.
        * This is to remove the noise from the Franka Panda's gripping time.
* For the remaining portions of active blocks, it ensures `Curvature` is constant (set to its value at the original start of the block) and `Position_cm` reflects original values.
* Recalculates `Curvature_Active`.

The output files from this stage are saved in the `csv_data/cleaned/` directory with the naming convention:
`cleaned_<CURVATURE_STR>[TEST_ID].csv`

Example:
* `cleaned_0_01[test 1].csv`

These `cleaned_*.csv` files are then used as input for the GPR model training scripts (e.g., `RQ_GPR_train.py`).