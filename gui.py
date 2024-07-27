import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QCheckBox, QScrollArea)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from plotting import get_plot_figure

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, figure=None, width=5, height=4, dpi=100):
        if figure is None:
            fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        else:
            fig = figure
        super().__init__(fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vector Database Benchmarking Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabs.addTab(self.tab1, "Results")
        self.tabs.addTab(self.tab2, "Data Generation")
        self.tabs.addTab(self.tab3, "Settings")

        self.initUI()

    def initUI(self):
        self.initTab1()
        self.initTab2()
        self.initTab3()

    def initTab1(self):
        layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scrollContent = QWidget()
        self.scrollLayout = QVBoxLayout(scrollContent)

        self.dataset_files = ["./result/result_small.json", "./result/result.json"]
        self.dataset_names = ["Small Dataset", "Large Dataset"]
        self.metrics = ['create_time', 'insert_time', 'similarity_time', 'size']

        for i, metric in enumerate(self.metrics):
            metric_layout = QVBoxLayout()

            button_layout = QHBoxLayout()
            for j, (dataset, dataset_name) in enumerate(zip(self.dataset_files, self.dataset_names)):
                dataset_button = QPushButton(f"{dataset_name} - {metric.replace('_', ' ').title()}")
                dataset_button.setCheckable(True)
                dataset_button.setChecked(False)  # Initially unchecked
                dataset_button.clicked.connect(lambda _, m=metric, d=dataset, btn=dataset_button: self.updatePlot(m, d, btn))
                button_layout.addWidget(dataset_button)
                if j == 0:
                    default_dataset = dataset
                    first_plot_button = dataset_button
                    first_metric_layout = metric_layout  # Track the correct layout for the first plot

            metric_layout.addLayout(button_layout)
            fig = get_plot_figure(metric, default_dataset)
            plot_canvas = PlotCanvas(scrollContent, figure=fig, width=12, height=8)
            metric_layout.addWidget(plot_canvas)
            plot_canvas.setVisible(False)  # Initially hidden

            self.scrollLayout.addLayout(metric_layout)

        first_plot_button.setChecked(True)
        self.updatePlot(self.metrics[0], self.dataset_files[0], first_plot_button)

        scroll.setWidget(scrollContent)
        layout.addWidget(scroll)
        self.tab1.setLayout(layout)

        # Ensure the first plot is visible in the correct location
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).layout()
            if widget == first_metric_layout:
                plot_canvas = widget.itemAt(1).widget()
                plot_canvas.setVisible(True)
            
    def initTab2(self):
        layout = QVBoxLayout()

        self.dataset_name = QLineEdit()
        self.dataset_name.setPlaceholderText("Dataset Name")
        layout.addWidget(self.dataset_name)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Dataset Path")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browsePath)
        layout.addWidget(self.dataset_path)
        layout.addWidget(browse_button)

        self.num_rows = QLineEdit()
        self.num_rows.setPlaceholderText("Number of Rows")
        layout.addWidget(self.num_rows)

        self.vector_dim = QLineEdit()
        self.vector_dim.setPlaceholderText("Vector Dimension")
        layout.addWidget(self.vector_dim)

        self.clustered = QCheckBox("Clustered")
        layout.addWidget(self.clustered)

        generate_button = QPushButton("Generate Data")
        generate_button.clicked.connect(self.generateData)
        layout.addWidget(generate_button)

        self.tab2.setLayout(layout)

    def initTab3(self):
        layout = QVBoxLayout()

        self.db_folder = QLineEdit()
        self.db_folder.setPlaceholderText("Database Folder")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browseDBFolder)
        layout.addWidget(self.db_folder)
        layout.addWidget(browse_button)

        self.pgvector_checkbox = QCheckBox("PGVector")
        self.milvus_checkbox = QCheckBox("Milvus")
        self.qdrant_checkbox = QCheckBox("Qdrant")
        layout.addWidget(self.pgvector_checkbox)
        layout.addWidget(self.milvus_checkbox)
        layout.addWidget(self.qdrant_checkbox)

        self.pgvector_settings = QLineEdit()
        self.pgvector_settings.setPlaceholderText("PGVector Settings (username, db_name, etc.)")
        layout.addWidget(self.pgvector_settings)

        self.qdrant_settings = QLineEdit()
        self.qdrant_settings.setPlaceholderText("Qdrant Settings (db_path, etc.)")
        layout.addWidget(self.qdrant_settings)

        self.tab3.setLayout(layout)

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
                        new_plot_canvas = PlotCanvas(parent=plot_canvas.parent(), figure=fig, width=12, height=8)
                        
                        # Remove the old plot canvas
                        widget.removeWidget(plot_canvas)
                        plot_canvas.setParent(None)  # This ensures the widget is properly destroyed

                        # Add the new plot canvas
                        widget.addWidget(new_plot_canvas)

                        new_plot_canvas.setVisible(True)
                    else:
                        plot_canvas.setVisible(False)                        

    def browsePath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.dataset_path.setText(path)

    def browseDBFolder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.db_folder.setText(path)

    def generateData(self):
        dataset_name = self.dataset_name.text()
        dataset_path = self.dataset_path.text()
        num_rows = int(self.num_rows.text())
        vector_dim = int(self.vector_dim.text())
        clustered = self.clustered.isChecked()

        # Placeholder: generate data logic
        print(f"Generated dataset {dataset_name} at {dataset_path} with {num_rows} rows and {vector_dim} dimensions. Clustered: {clustered}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
