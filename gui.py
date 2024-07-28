import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QCheckBox,
    QScrollArea, QLabel, QSpinBox)
# from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
import matplotlib.pyplot as plt
from plotting import get_plot_figure
from benchmark import Benchmark
from data_generation import generate_dataset
import pandas as pd


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, figure=None, width=5, height=4, dpi=100):
        if figure is None:
            fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        else:
            fig = figure
        super().__init__(fig)
        self.setParent(parent)


class PlotData(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, data, x_index=0, y_index=1, num_samples=100):
        self.ax.clear()
        sampled_data = data.sample(n=num_samples)
        self.ax.scatter(sampled_data.iloc[:, x_index],
                        sampled_data.iloc[:, y_index])
        self.ax.set_xlabel(f"Dimension {x_index}")
        self.ax.set_ylabel(f"Dimension {y_index}")
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.datasets_result_files = ["./result/small_dataset_result.json",
                                      "./result/large_dataset_result.json"]
        self.datasets_files = ["./data/small_dataset/data.csv",
                               "./data/large_dataset/data.csv"]
        self.dataset_names = ["Small Dataset", "Large Dataset"]
        self.metrics = ['create_time', 'insert_time',
                        'similarity_time', 'size']

        self.setWindowTitle("Vector Database Benchmarking Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabs.addTab(self.tab1, "Results")
        self.tabs.addTab(self.tab2, "Data Generation")
        self.tabs.addTab(self.tab3, "Benchmark Test")

        self.initUI()

    def initUI(self):
        self.initTab1()
        self.initTab2()
        self.initTab3()

    def initTab1(self):
        self.tab1_layout = QVBoxLayout()
        self.tab1_scroll = QScrollArea()
        self.tab1_scroll.setWidgetResizable(True)
        self.scrollContent = QWidget()
        self.scrollLayout = QVBoxLayout(self.scrollContent)
        self.initTab1Content()
        self.tab1.setLayout(self.tab1_layout)

    def initTab1Content(self):
        for i, metric in enumerate(self.metrics):
            metric_layout = QVBoxLayout()

            button_layout = QHBoxLayout()
            for j, (dataset, dataset_name) in enumerate(
                 zip(self.datasets_result_files, self.dataset_names)):
                dataset_button = QPushButton(
                    f"{dataset_name} - {metric.replace('_', ' ').title()}")
                dataset_button.setCheckable(True)
                dataset_button.setChecked(False)  # Initially unchecked
                dataset_button.clicked.connect(
                    lambda _, m=metric, d=dataset,
                    btn=dataset_button: self.updatePlot(m, d, btn))
                button_layout.addWidget(dataset_button)
                if i == 0 and j == 0:
                    default_dataset = dataset
                    first_plot_button = dataset_button
                    # Track the correct layout for the first plot
                    first_metric_layout = metric_layout

            metric_layout.addLayout(button_layout)
            fig = get_plot_figure(metric, default_dataset)
            plot_canvas = PlotCanvas(self.scrollContent,
                                     figure=fig, width=12, height=8)
            plot_canvas.setVisible(False)  # Initially hidden
            metric_layout.addWidget(plot_canvas)
            self.scrollLayout.addLayout(metric_layout)

        first_plot_button.setChecked(True)
        self.updatePlot(self.metrics[0], self.datasets_result_files[0],
                        first_plot_button)

        self.tab1_scroll.setWidget(self.scrollContent)
        self.tab1_layout.addWidget(self.tab1_scroll)

        # Ensure the first plot is visible in the correct location
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).layout()
            if widget == first_metric_layout:
                plot_canvas = widget.itemAt(1).widget()
                plot_canvas.setVisible(True)

    def initTab2(self):
        layout = QVBoxLayout()

        # Dataset Name
        hbox_name = QHBoxLayout()
        self.dataset_name = QLineEdit()
        self.dataset_name.setPlaceholderText("Dataset Name")
        hbox_name.addWidget(QLabel("Dataset Name:"))
        hbox_name.addWidget(self.dataset_name)
        layout.addLayout(hbox_name)

        # Dataset Path
        hbox_path = QHBoxLayout()
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Dataset Path")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browsePath)
        hbox_path.addWidget(QLabel("Dataset Path:"))
        hbox_path.addWidget(self.dataset_path)
        hbox_path.addWidget(browse_button)
        layout.addLayout(hbox_path)

        # Number of Rows
        self.num_rows = QLineEdit("1000")
        hbox_rows = QHBoxLayout()
        hbox_rows.addWidget(QLabel("Number of rows:"))
        hbox_rows.addWidget(self.num_rows)
        layout.addLayout(hbox_rows)

        # Number of Dimensions
        self.vector_dim = QLineEdit("100")
        hbox_dims = QHBoxLayout()
        hbox_dims.addWidget(QLabel("Number of dimensions:"))
        hbox_dims.addWidget(self.vector_dim)
        layout.addLayout(hbox_dims)

        # Clustered
        self.clustered = QCheckBox("Clustered")
        self.clustered.setChecked(True)
        layout.addWidget(self.clustered)

        # Generate Parquet
        self.generate_parquet = QCheckBox("Generate Parquet")
        layout.addWidget(self.generate_parquet)

        # Add to Datasets List
        self.add_to_datasets = QCheckBox("Add to Datasets List")
        self.add_to_datasets.setChecked(True)
        layout.addWidget(self.add_to_datasets)

        # Generate Data Button
        generate_button = QPushButton("Generate Data")
        generate_button.clicked.connect(self.generateData)
        layout.addWidget(generate_button)

        # Visualization
        self.visualization_layout = QVBoxLayout()
        self.plot_data = PlotData(self, width=5, height=4, dpi=100)
        self.visualization_layout.addWidget(self.plot_data)

        # Spinboxes for visualization
        hbox_spin1 = QHBoxLayout()
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 99)
        self.spin_x.setValue(0)
        hbox_spin1.addWidget(QLabel("X Dimension:"))
        hbox_spin1.addWidget(self.spin_x)
        layout.addLayout(hbox_spin1)

        hbox_spin2 = QHBoxLayout()
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 99)
        self.spin_y.setValue(1)
        hbox_spin2.addWidget(QLabel("Y Dimension:"))
        hbox_spin2.addWidget(self.spin_y)
        layout.addLayout(hbox_spin2)

        hbox_samples = QHBoxLayout()
        self.num_samples = QLineEdit("100")
        hbox_samples.addWidget(QLabel("Number of samples:"))
        hbox_samples.addWidget(self.num_samples)
        replot_button = QPushButton("Replot")
        replot_button.clicked.connect(self.replotData)
        hbox_samples.addWidget(replot_button)
        layout.addLayout(hbox_samples)

        layout.addLayout(self.visualization_layout)
        self.tab2.setLayout(layout)

    def generateData(self):
        dataset_name = self.dataset_name.text()
        dataset_path = self.dataset_path.text()
        num_rows = int(self.num_rows.text())
        num_dimensions = int(self.vector_dim.text())
        clustered = self.clustered.isChecked()
        parquet = self.generate_parquet.isChecked()

        # Create the dataset directory
        name = dataset_name.replace(" ", "_").lower()
        full_path = f"{dataset_path}/{name}"
        os.makedirs(full_path, exist_ok=True)

        # Generate the dataset
        generate_dataset(num_vectors=num_rows,
                         num_dimensions=num_dimensions,
                         folder_path=full_path,
                         cluster=clustered, parquet=parquet)

        # Add to datasets list if checked
        if self.add_to_datasets.isChecked() and\
                dataset_name not in self.dataset_names:
            self.datasets_files.append(f"{full_path}/data.csv")
            self.dataset_names.append(dataset_name)
            # Update the datasets section in Tab 3
            self.addDatasetToTab3(dataset_name)

        # Load the generated data for visualization
        self.generated_data = pd.read_csv(f"{full_path}/data.csv")
        self.updateVisualization()

    def addDatasetToTab3(self, dataset_name):
        checkbox = QCheckBox(dataset_name)
        checkbox.setChecked(True)
        self.dataset_checkboxes.append(checkbox)
        # Add the new dataset checkbox to the layout
        datasets_section_layout = self.tab3.layout().itemAt(0).layout()
        datasets_section_layout.insertWidget(len(self.dataset_checkboxes) - 1,
                                             checkbox)

    def updateVisualization(self):
        num_samples = int(self.num_samples.text())
        x_index = self.spin_x.value()
        y_index = self.spin_y.value()
        self.plot_data.plot(self.generated_data, x_index, y_index, num_samples)

    def replotData(self):
        self.updateVisualization()

    def initTab3(self):
        layout = QVBoxLayout()

        # Datasets Section
        datasets_section = QVBoxLayout()
        datasets_label = QLabel("Datasets")
        datasets_section.addWidget(datasets_label)

        self.dataset_checkboxes = []
        for dataset_name in self.dataset_names:
            checkbox = QCheckBox(dataset_name)
            checkbox.setChecked(True)
            self.dataset_checkboxes.append(checkbox)
            datasets_section.addWidget(checkbox)

        hbox_new_dataset = QHBoxLayout()
        self.new_dataset_name = QLineEdit()
        self.new_dataset_name.setPlaceholderText("New Dataset Name")
        hbox_new_dataset.addWidget(self.new_dataset_name)
        datasets_section.addLayout(hbox_new_dataset)

        hbox_new_dataset_path = QHBoxLayout()
        self.new_dataset_path = QLineEdit()
        self.new_dataset_path.setPlaceholderText("New Dataset Path")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browseNewDatasetPath)
        hbox_new_dataset_path.addWidget(self.new_dataset_path)
        hbox_new_dataset_path.addWidget(browse_button)
        datasets_section.addLayout(hbox_new_dataset_path)

        add_dataset_button = QPushButton("Add Dataset")
        add_dataset_button.clicked.connect(self.addDataset)
        datasets_section.addWidget(add_dataset_button)

        layout.addLayout(datasets_section)

        # Frameworks Section
        frameworks_section = QVBoxLayout()
        frameworks_label = QLabel("Frameworks")
        frameworks_section.addWidget(frameworks_label)

        self.pgvector_checkbox = QCheckBox("PGVector")
        self.pgvector_checkbox.stateChanged.connect(self.togglePGVector)
        frameworks_section.addWidget(self.pgvector_checkbox)

        self.pg_dbname = QLineEdit('postgres')
        self.pg_dbname.setPlaceholderText("PGVector DB Name")
        self.pg_username = QLineEdit('billyslim')
        self.pg_username.setPlaceholderText("PGVector Username")
        self.pg_password = QLineEdit('')
        self.pg_password.setPlaceholderText("PGVector Password")
        frameworks_section.addWidget(self.pg_dbname)
        frameworks_section.addWidget(self.pg_username)
        frameworks_section.addWidget(self.pg_password)
        self.pg_dbname.setEnabled(False)
        self.pg_username.setEnabled(False)
        self.pg_password.setEnabled(False)

        self.milvus_checkbox = QCheckBox("Milvus")
        self.milvus_checkbox.stateChanged.connect(self.toggleMilvus)
        frameworks_section.addWidget(self.milvus_checkbox)

        hbox_milvus = QHBoxLayout()
        self.milvus_db_path = QLineEdit('milvus_db/milvus_demo.db')
        self.milvus_db_path.setPlaceholderText("Milvus DB Path")
        browse_milvus_button = QPushButton("Browse")
        browse_milvus_button.clicked.connect(self.browseMilvusPath)
        hbox_milvus.addWidget(self.milvus_db_path)
        hbox_milvus.addWidget(browse_milvus_button)
        frameworks_section.addLayout(hbox_milvus)
        self.milvus_db_path.setEnabled(False)
        browse_milvus_button.setEnabled(False)

        self.qdrant_checkbox = QCheckBox("Qdrant")
        self.qdrant_checkbox.stateChanged.connect(self.toggleQdrant)
        frameworks_section.addWidget(self.qdrant_checkbox)

        hbox_qdrant = QHBoxLayout()
        self.qdrant_db_path = QLineEdit('./qdrant_data')
        self.qdrant_db_path.setPlaceholderText("Qdrant DB Path")
        browse_qdrant_button = QPushButton("Browse")
        browse_qdrant_button.clicked.connect(self.browseQdrantPath)
        hbox_qdrant.addWidget(self.qdrant_db_path)
        hbox_qdrant.addWidget(browse_qdrant_button)
        frameworks_section.addLayout(hbox_qdrant)
        self.qdrant_db_path.setEnabled(False)
        browse_qdrant_button.setEnabled(False)

        layout.addLayout(frameworks_section)

        # Run Tests Section
        run_tests_section = QVBoxLayout()
        run_tests_label = QLabel("Results")
        run_tests_section.addWidget(run_tests_label)

        hbox_result = QHBoxLayout()
        self.result_folder_path = QLineEdit('./result/')
        self.result_folder_path.setPlaceholderText("Result Folder Path")
        browse_result_button = QPushButton("Browse")
        browse_result_button.clicked.connect(self.browseResultFolder)
        hbox_result.addWidget(self.result_folder_path)
        hbox_result.addWidget(browse_result_button)
        run_tests_section.addLayout(hbox_result)

        run_tests_button = QPushButton("Run Tests")
        run_tests_button.clicked.connect(self.runTests)
        run_tests_section.addWidget(run_tests_button)

        layout.addLayout(run_tests_section)

        self.tab3.setLayout(layout)

    def browseNewDatasetPath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.new_dataset_path.setText(path)

    def addDataset(self):
        dataset_name = self.new_dataset_name.text()
        dataset_path = self.new_dataset_path.text()
        if dataset_name and dataset_path:
            checkbox = QCheckBox(dataset_name)
            checkbox.setChecked(True)
            self.dataset_checkboxes.append(checkbox)
            self.dataset_names.append(dataset_name)
            self.datasets_files.append(f"{dataset_path}/data.csv")
            self.tab3.layout().itemAt(0).layout()\
                .insertWidget(len(self.dataset_checkboxes) - 1, checkbox)

    def toggleMilvus(self):
        state = self.milvus_checkbox.isChecked()
        self.milvus_db_path.setEnabled(state)
        # Ensure the Milvus browse button is enabled/disabled correctly
        for i in range(self.tab3.layout().count()):
            item = self.tab3.layout().itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    widget = item.itemAt(j).widget()
                    if isinstance(widget, QPushButton) and\
                        widget.text() == "Browse" and\
                            widget.parent() == self.milvus_db_path:
                        widget.setEnabled(state)

    def toggleQdrant(self):
        state = self.qdrant_checkbox.isChecked()
        self.qdrant_db_path.setEnabled(state)
        # Ensure the Qdrant browse button is enabled/disabled correctly
        for i in range(self.tab3.layout().count()):
            item = self.tab3.layout().itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    widget = item.itemAt(j).widget()
                    if isinstance(widget, QPushButton) and\
                        widget.text() == "Browse" and\
                            widget.parent() == self.qdrant_db_path:
                        widget.setEnabled(state)

    def togglePGVector(self):
        state = self.pgvector_checkbox.isChecked()
        self.pg_dbname.setEnabled(state)
        self.pg_username.setEnabled(state)
        self.pg_password.setEnabled(state)

    def browseMilvusPath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.milvus_db_path.setText(path)

    def browseQdrantPath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.qdrant_db_path.setText(path)

    def browseResultFolder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.result_folder_path.setText(path)

    def runTests(self):
        new_dataset_names = []
        new_dataset_results = []
        for checkbox, dataset_name, dataset_file in zip(
             self.dataset_checkboxes, self.dataset_names, self.datasets_files):
            if checkbox.isChecked():
                test_csv_path = dataset_file.replace("data.csv", "test.csv")
                result_file = f"{self.result_folder_path.text()}/" +\
                    f"{checkbox.text().replace(' ', '_').lower()}_result.json"
                pgname = self.pg_dbname.text() if self\
                    .pgvector_checkbox.isChecked() else ''
                pgusername = self.pg_username.text() if self\
                    .pgvector_checkbox.isChecked() else ''
                pgpassword = self.pg_password.text() if self\
                    .pgvector_checkbox.isChecked() else ''
                milvusdb_path = self.milvus_db_path.text() if self\
                    .milvus_checkbox.isChecked() else ''
                qdrantdb_path = self.qdrant_db_path.text() if self\
                    .qdrant_checkbox.isChecked() else ''

                Benchmark(
                    csv_path=dataset_file, test_csv_path=test_csv_path,
                    result_file=result_file, pg_dbname=pgname,
                    pg_username=pgusername, pg_password=pgpassword,
                    milvus_db_path=milvusdb_path, qdrant_db_path=qdrantdb_path
                )
                new_dataset_names.append(dataset_name)
                new_dataset_results.append(result_file)
        self.datasets_result_files = new_dataset_results
        self.dataset_names = new_dataset_names
        # Clear the existing layout of Tab 1 before reinitializing it
        self.clearLayout(self.scrollLayout)
        self.initTab1Content()

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget() is not None:
                    child.widget().deleteLater()
                elif child.layout() is not None:
                    self.clearLayout(child.layout())

    def updatePlot(self, metric, dataset, sender_button):
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).layout()
            if widget:
                buttons_layout = widget.itemAt(0).layout()
                found_plot = False
                if buttons_layout:
                    for j in range(buttons_layout.count()):
                        button = buttons_layout.itemAt(j).widget()
                        if isinstance(button, QPushButton):
                            button.setChecked(button == sender_button)
                        if button == sender_button:
                            found_plot = True
                plot_canvas = widget.itemAt(1).widget()
                if isinstance(plot_canvas, PlotCanvas):
                    plot_canvas.setVisible(False)
                    if found_plot:
                        # Create a new figure and PlotCanvas
                        fig = get_plot_figure(metric, dataset)
                        new_plot_canvas = PlotCanvas(
                            parent=plot_canvas.parent(),
                            figure=fig, width=12, height=8)
                        # Remove the old plot canvas
                        widget.removeWidget(plot_canvas)
                        # This ensures the widget is properly destroyed
                        plot_canvas.setParent(None)

                        # Add the new plot canvas
                        widget.addWidget(new_plot_canvas)

                        new_plot_canvas.setVisible(True)
                    else:
                        plot_canvas.setVisible(False)

    def browsePath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.dataset_path.setText(path)

    # def browseDBFolder(self):
    #     path = QFileDialog.getExistingDirectory(self, "Select Directory")
    #     self.db_folder.setText(path)

    # def generateData(self):
    #     dataset_name = self.dataset_name.text()
    #     dataset_path = self.dataset_path.text()
    #     num_rows = int(self.num_rows.text())
    #     vector_dim = int(self.vector_dim.text())
    #     clustered = self.clustered.isChecked()

    #     # Placeholder: generate data logic
    #     print(f"Generated dataset {dataset_name} at " +
    #           f"{dataset_path} with {num_rows} rows and " +
    #           f"{vector_dim} dimensions. Clustered: {clustered}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
