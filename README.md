# Machine Learning Methods in Network Similarity Detection—Master's Thesis

## Overview

This application provides a comprehensive suite of tools for analyzing 
graph similarity based on graphlet distributions, using various machine learning 
and statistical methods. 
The application consists of three primary subsystems:

1. **Process Files**: 

Process graph files using [ORCA](https://github.com/thocevar/orca/tree/master) to get graphlet distributions 
to compute similarity using multiple methods:
   - Hellinger Distance
   - NetSimile
   - Resnet
   - Kolmogorov-Smirnov 2-sample test

2. **Training Model**: 

Train machine learning models for graph similarity prediction:
   - Random Forest Classifier
   - Multi-Layer Perceptron (MLP)
   - Hyperparameter tuning capabilities

3. **Predict Similarity**: 

Apply trained models to predict similarity between graphs

## System Requirements

- Python 3.10 or higher
- Windows/Linux/MacOS
- Storage space: 2GB minimum (for application and working files)

## Installation

### Option 1: Using the Executable (Recommended for End Users on Windows)

1. Download the zip file from the GitHub releases page
2. Extract the zip file to your preferred location
3. Run the executable file to start the application

### Option 2: Building from Source

1. Clone the repository:
   ```
   git clone https://github.com/MarianFigula/graph-similarity-detection.git
   cd graph-similarity-detection
   ```

2. Set up a Python virtual environment (recommended):
   ```
   python -m venv venv
   ```
   
   Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install SciPy hooks for PyInstaller:
   ```
   python install_hooks.py
   ```

5. Build the executable:
   ```
   pyinstaller --clean app.spec
   ```

6. The executable will be created in the `dist` directory

## Dependencies

The application relies on numerous Python libraries for its functionality. 
The complete list of dependencies can be found in the `requirements.txt` file.

## Usage Guide

The application is structured to make it easy to navigate and use.
Every subsystem has tutorial text which can be found in 
application in the top right corner of the screen (button with question mark).

### Subsystem 1: Processing Files and Labeling

This subsystem allows you to analyze the similarity between graphs using multiple methods.

1. **Input Data**: 
   - Choose directory containing graph files
   - The Input file describes the network in a simple text format. 
   - The first line contains two integers n and e - the number of nodes and edges. 
   - The following e lines describe undirected edges with space-separated ids of their endpoints. 
   - Node ids should be between 0 and n-1. See the example file in the `input` directory.
   - Application can also process already generated `.out` files from ORCA.

2. **Processing Files and Labeling**:
   - Launch the application
   - Navigate to the "Process Files" section
   - Select the graphs you wish to compare
   - Choose one or more similarity methods:
     - **Hellinger Distance**: Compares graphlet distributions 
     - **NetSimile**: Uses feature extraction with canberra distance
     - **Resnet**: Approach for comparing graphlet distributions as images
     - **Kolmogorov-Smirnov 2-sample test**: Statistical test for graphlet distribution comparison
   - Choose weight for each method

3. **Results**:
   - The application will generate similarity scores for each selected method, generate thresholds using K-Means and make 
   final binary similarity decisions as a sum of weighted methods with their labels
   - Results can be exported as CSV or visualized directly in the application

### Subsystem 2: Model Training

This subsystem allows you to train machine learning models for graph similarity prediction.

1. **Training Data**:
   - Prepare a dataset of graph pairs with known similarity labels and their graphlet distributions—if you don't have it, use the [Processing and Labeling subsystem](#subsystem-1-processing-files-and-labeling)

2. **Training Models**:
   - Navigate to the "Training Model" section
   - Select the model type:
     - **Random Forest Classifier**: Ensemble learning method
     - **Multi-Layer Perceptron (MLP)**: Neural network approach
   - Configure hyperparameters:
     - For Random Forest: number of trees, max depth, etc.
     - For MLP: hidden layer sizes, learning rate, early stopping, etc.
   - Start the training process

3. **Model Evaluation**:
   - The application will display performance metrics
   - Visualize model performance through ROC curves and confusion matrices

4. **Model Storage**:
   - Trained models are automatically saved in `MachineLearningData/saved_models` and can be accessed later in [Predict Similarity subsystem](#subsystem-3-predict-similarity)

### Subsystem 3: Predict Similarity

This subsystem allows you to use trained models to predict similarity between new graphs.

1. **Model Selection**:
   - Navigate to the "Predict Similarity" section
   - Select a previously trained model from the drop-down menu

2. **Graph Selection**:
   - Choose the graph pairs you want to analyze
   - If you want to compare between 2 groups of graphs, check the "Compare between two files" checkbox

3. **Prediction**:
   - Run the prediction process
   - View similarity scores and binary similarity decisions
   - 
4. **Export Results**:
   - Export results as CSV or visualize directly in the application
   - Results will be stored in `MachineLearningData/predictions`

## License

This application is released under the [MIT License](LICENSE).
