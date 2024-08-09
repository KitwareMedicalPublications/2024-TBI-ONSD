# A few notes about the imports here
# PySide2 must be imported before pyqtgraph, so that pyqtgraph knows
# to use PySide2 as a backend and not go look for PyQt5.
# However, something in the deep_learning import is causing it to still search for
# PyQt5 if deep_learning is imported before pyqtgraph. So the order of the imports
# here shouldn't change much or things may break.

import os
from pathlib import Path
from PySide2.QtGui import QPixmap, QImage, QPainter
from PySide2.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QHBoxLayout,
    QWidget,
    QVBoxLayout,
)
from PySide2.QtCore import Signal, QThread, Qt, QTimer
import pyqtgraph as pg
from multiprocessing import Queue, Process

import argparse
import sys
import signal
import itk
import pyigtl
import sys
import numpy as np
import time
import tbitk.ai.deep_learning as dl
import torch
import matplotlib
from tbitk.ocularus import aggregate_onsds


class MainWindow(QMainWindow):
    """
    Main window for the Qt GUI.

    Contains the source video frames, mask frames, and an optional plot of
    the onsd prediction over time.

    Attributes
    ----------
    plot_onsd : bool
        When set, plots the onsd over time.
        Note that this results in lower fps
    separate_source_and_mask : bool
        Create a separate widget for the mask. If false, displays the
        mask (with reduced opacity) on top of the source
    mask_cmap : matplotlib.colors.ColorMap
        cmap to apply to the mask
    title : str
        Title of the window
    """
    def __init__(
        self,
        q,
        model_path,
        plot_onsd=True,
        separate_source_and_mask=True,
        normalize=False,
        mask_cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "maskcmap", ["black", "yellow", "pink"]
        ),
        encoder_name="resnet34"
    ):
        """
        Parameters
        ----------
        q : multiprocessing.Queue
            Queue to pass on to `getImageThread`
        model_path : str
            Path to the model to use for inference
        plot_onsd : bool
            Whether or not to plot the aggregated onsd over time
        separate_source_and_mask : bool
            Flag to separate the source and mask widget. Default is True
        normalize : bool
            Passed on to `getImageThread`. Normalize the input source if True
        mask_cmap : matplotlib.colors.ColorMap
            Colormap to apply to the mask
        """
        super(MainWindow, self).__init__()

        # TODO: Properly give credit. Following lines / approach are taken from
        # https://github.com/lassoan/pyigtl/blob/master/pyigtl/comm.py#L186
        self._previous_signal_handlers = {}
        self._previous_signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
        self._previous_signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)

        self.plot_onsd = plot_onsd
        self.separate_source_and_mask = separate_source_and_mask
        self.mask_cmap = mask_cmap

        # Initializing our worker thread.
        self._thread = getImageThread(q, model_path, plot_onsd, normalize=normalize, encoder_name=encoder_name)
        # Connect the worker threads signal to our
        # update_display method
        self._thread.signal.connect(self.update_display)

        self.title = "Image Viewer"
        self.setWindowTitle(self.title)

        # Set up source widget
        # TODO: strange to have this accept a QImage argument
        init_pixmap = self.array_to_pixmap(
            np.zeros((512, 512)), 1, QImage.Format_Grayscale8
        )
        self._source_widget = QLabel(self)
        self._source_widget.setFixedSize(512, 512)
        self._source_widget.setPixmap(init_pixmap)

        # Add source image widget and a title to a layout.
        # Title will be above image, so it gets added first.
        self._source_with_title_layout = QVBoxLayout()
        self._source_title_label = QLabel("Source")
        self._source_title_label.setAlignment(Qt.AlignCenter)
        self._source_with_title_layout.addWidget(self._source_title_label)
        self._source_with_title_layout.addWidget(self._source_widget)

        if self.separate_source_and_mask:
            # Add mask widget, if present
            self._mask_widget = QLabel(self)
            self._mask_widget.setFixedSize(512, 512)
            self._mask_widget.setPixmap(init_pixmap)
            # Same process for adding a title
            self._mask_with_title_layout = QVBoxLayout()
            self._mask_title_label = QLabel("Mask")
            self._mask_title_label.setAlignment(Qt.AlignCenter)
            self._mask_with_title_layout.addWidget(self._mask_title_label)
            self._mask_with_title_layout.addWidget(self._mask_widget)

            # Now we have 2 layouts, one for the image and its title,
            # the other for the mask and its title. We'll stack these
            # layous horizontally. Source image + title on the left
            # mask + title on the right
            self._image_layout = QHBoxLayout()
            self._image_layout.addLayout(self._source_with_title_layout)
            self._image_layout.addLayout(self._mask_with_title_layout)
        else:
            self._source_with_title_layout.setAlignment(Qt.AlignCenter)

        # Get the plot widget ready. Here we'll plot the onsd predictions
        # over time. We'll include error bars.
        if self.plot_onsd:
            self._graph_widget = pg.PlotWidget()
            self._graph_widget.setLimits(yMin=0)
            self._graph_widget.setYRange(3, 9.5, padding=0)
            self._graph_widget.setLabel(axis="left", text="Predicted ONSD (mm)")
            self._graph_widget.setLabel(axis="bottom", text="Number of Collected Frames")
            self._onsd_estimate_line = self._graph_widget.plot()
            self._error_bar = pg.ErrorBarItem()
            self._graph_widget.addItem(self._error_bar)

            # Add a title to the plot by using a layout, just like image and
            # mask layouts above.
            self._plot_with_title_layout = QVBoxLayout()
            self._plot_title_label = QLabel("ONSD Measurement vs Number of Collected Frames")
            self._plot_title_label.setAlignment(Qt.AlignCenter)
            self._plot_with_title_layout.addWidget(self._plot_title_label)
            self._plot_with_title_layout.addWidget(self._graph_widget)

        # Main layout contains at most two things stacked vertically
        # 1.) The layout containing the either image and mask sources and their titles,
        #     or just the layout for the source image and its title
        # 2.) The layout containing the plot and its title (if we are plotting)
        # Initialize the main layout here
        self._main_layout = QVBoxLayout()
        if self.separate_source_and_mask:
            self._main_layout.addLayout(self._image_layout)
        else:
            self._main_layout.addLayout(self._source_with_title_layout)

        if self.plot_onsd:
            self._main_layout.addLayout(self._plot_with_title_layout)

        # This is for plotting. Keep track of x and y data.
        # "connect" list is for line breaks. self._connect[i] is 1 if
        # there should be a line connecting (self._x[i], self._y[i]) to
        # (self._x[i+1], self._y[i+1]). If previous_frame_had_prediction is true
        # and we have a prediction for the current frame, we add a line break.
        self._x = []
        self._y = []
        self._connect = []
        self._previous_frame_had_prediction = False

        # Add layout to the central widget
        central_widget = QWidget()
        central_widget.setLayout(self._main_layout)
        self.setCentralWidget(central_widget)

        # Start worker thread
        self._thread.start()


    def _signal_handler(self, signalnum, stackframe):
        # TODO: Give credit to https://github.com/lassoan/pyigtl/blob/master/pyigtl/comm.py#L186
        # Following lines are similar.
        self.close()

        # Restore previous handler and re-send signal
        previous_handler = self._previous_signal_handlers[signalnum]
        signal.signal(signalnum, previous_handler)
        os.kill(os.getpid(), signalnum)

    def apply_matplotlib_color_map_to_arr(self, arr, cmap, norm_factor=None):
        """
        Apply a matplotlib colormap to a numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array to apply the colormap to
        cmap : matplotlib.colors.ColorMap
            Colormap to apply
        norm_factor : float
            Normalization factor to apply before applying the colormap

        Returns
        -------
        np.ndarray
        """

        arr = arr.astype(np.float32)
        if norm_factor is not None:
            arr /= norm_factor

        return cmap(arr)

    def array_to_qimage(self, arr, bytes_per_pixel, format_):
        """
        Convenience function to convert a numpy array to a QImage.

        Parameters
        ----------
        arr : np.ndarray
            Array to convert
        bytes_per_pixel : int
            The number of bytes per pixel
        format_ : PySide2.QtGui.QImage.Format
            Format to use when constructing the QImage.

        Returns
        -------
        PySide2.QtGui.QImage
        """

        return QImage(
            arr, arr.shape[1], arr.shape[0], bytes_per_pixel * arr.shape[1], format_
        )

    def array_to_pixmap(self, arr, bytes_per_pixel, format_):
        """
        Convenience function to convert a numpy array to a QPixmap.
        First converts to a QImage, then to a QPixmap

        Parameters
        ----------
        arr : np.ndarray
            Array to convert
        bytes_per_pixel : int
            The number of bytes per pixel
        format_ : PySide2.QtGui.QImage.Format
            Format to use when constructing the QPixmap.

        Returns
        -------
        PySide2.QtGui.QPixmap
        """

        arr = (arr * 255).astype(np.uint8)
        return QPixmap(self.array_to_qimage(arr, bytes_per_pixel, format_))

    def ultrasound_image_to_pixmap(self, img, bytes_per_pixel, format_):
        """
        Convenience function to convert a itk ultrasound image
        straight to a QPixmap.

        Parameters
        ----------
        img : itk.Image[itk.F, 2]
            Image to convert
        bytes_per_pixel : int
            The number of bytes per pixel
        format_ : PySide2.QtGui.QImage.Format
            Format to use when constructing the QPixmap.

        Returns
        -------
        PySide2.QtGui.QPixmap
        """

        arr = itk.array_from_image(img)
        return self.array_to_pixmap(arr, bytes_per_pixel, format_)

    def closeEvent(self, event):
        """
        Executes after the GUI is closed. Stops the child thread

        Parameters
        ----------
        event : QtCore.QEvent

        Returns
        -------
        None
        """

        self._thread.requestInterruption()
        self._thread.wait()
        super(QMainWindow, self).closeEvent(event)

    def update_display(self, img, mask, aggregated_onsd_pred, start_of_new_line):
        """
        Update the display with a new image, mask, and onsd prediction.

        Parameters
        ----------
        img : itk.Image[itk.F, 2]
            Image to display
        mask : itk.Image[itk.UC, 2]
            Mask to display
        aggregated_onsd_pred : float
            Aggregated onsd value to plot
        start_of_new_line : bool
            If plotting, whether or not the current onsd prediction is
            the start of a new line in the plot.
        Returns
        -------
        None
        """
        source_pixmap = self.ultrasound_image_to_pixmap(
            img, 1, QImage.Format_Grayscale8
        )
        # Going to do some extra processing to the mask pixmap
        mask_arr = itk.array_from_image(mask)
        indices_arr_zero = np.where(mask_arr == 0)
        mask_arr = self.apply_matplotlib_color_map_to_arr(
            mask_arr,
            self.mask_cmap,
            norm_factor=max(dl.EYE_PIXEL_VALUE, dl.NERVE_PIXEL_VALUE),
        )
        if not self.separate_source_and_mask:
            alphas = mask_arr[:, :, 3]
            alphas[indices_arr_zero] = 0
            alphas[alphas != 0] = 1 / 3
            mask_arr[:, :, 3] = alphas
        mask_pixmap = self.array_to_pixmap(mask_arr, 4, QImage.Format_RGBA8888)

        source_pixmap = source_pixmap.scaled(512, 512)
        mask_pixmap = mask_pixmap.scaled(512, 512)

        if self.separate_source_and_mask:
            self._source_widget.setPixmap(source_pixmap)
            self._mask_widget.setPixmap(mask_pixmap)
        else:
            painter = QPainter(source_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, mask_pixmap)
            painter.end()
            self._source_widget.setPixmap(source_pixmap)

        if aggregated_onsd_pred is not None:
            self._x.append(self._x[-1] + 1 if self._x else 0)
            self._y.append(aggregated_onsd_pred)
            if start_of_new_line and self._connect:
                self._connect[-1] = 0

            self._connect.append(1)
            self._onsd_estimate_line.setData(
                self._x, self._y, connect=np.array(self._connect)
            )

            # Get variance.
            if self._y:
                first_data_index = max(0, len(self._y) - 10)
                samples = self._y[first_data_index:]
                self._error_bar.setData(
                    x=np.array(self._x[-1]),
                    y=np.array(samples[-1]),
                    height=np.var(samples),
                    beam=1 / 12,
                )


class getImageThread(QThread):
    """
    Thread to process video frames and run them through the model.

    Attributes
    ----------
    signal : PySide2.QtCore.Signal
        Signal to send back to the main window.
    get_onsd : bool
        Retrieve the onsd for each frame
    normalize : bool
        Divide the source image by 255 before processing.
    """
    signal = Signal(object, object, object, bool)

    def __init__(self, q, model_path, get_onsd, normalize=False, encoder_name="resnet34"):
        """
        Parameters
        ----------
        q : multiprocessing.Queue
            Queue containing source frames to process
        model_path : str
            Path to the model to use for inference
        get_onsd : bool
            Retrieve the onsd for each frame
        normalize : bool
            Divide the source image by 255 before processing.
        """
        QThread.__init__(self)
        self.get_onsd = get_onsd
        self.normalize = normalize
        self._onsds_so_far = []
        self._scores_so_far = []
        self._q = q

        self.model = dl.load_model(model_path, encoder_name=encoder_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


    def run(self):
        """
        Main function for the thread to execute.

        Pops source frames and the spacing off of the queue
        and runs the inference code. Sends the source frame, mask,
        aggregated onsd prediction, and a boolean flag for whether or not
        to start a new line break back to the main window.

        Returns
        -------
        None
        """

        previous_frame_had_prediction = False
        while not QThread.currentThread().isInterruptionRequested():
            if self._q.empty():
                # While the queue is empty, sleep for a bit then check again
                # We wont ever exit unless the user closes the gui
                time.sleep(0.1)
                continue

            npimg, spacing = self._q.get()
            if self.normalize:
                npimg = npimg / 255.0
                npimg = npimg.astype(np.float32)
            img = itk.image_from_array(npimg)
            img.SetSpacing(spacing)

            # Now feed into network and send to gui
            aggregated_onsd = None
            res_2d = dl.run_inference(self.model, img, self.device, get_onsd=self.get_onsd)
            if res_2d.onsd is not None:
                self._onsds_so_far.append(res_2d.onsd)
                self._scores_so_far.append(res_2d.score)
                aggregated_onsd = aggregate_onsds(
                    self._onsds_so_far, self._scores_so_far
                )
            self.signal.emit(
                img, res_2d.mask, aggregated_onsd, not previous_frame_had_prediction
            )

            previous_frame_had_prediction = res_2d.onsd is not None



def wait_and_add_image_to_queue(q, port, act_as_host, host, device_name, spacing=None):
    """
    Listens on a pyigtl connection for incoming image messages
    and adds them to a multiprocessing queue

    Parameters
    ----------
    q : multiprocessing.Queue
        Queue to add images onto
    port : int
        port to listen on
    act_as_host : bool
        Initialize the openigtlink connection as the host, not the client
    host : str
        host to connect to if `act_as_host` is false. Usually "127.0.0.1"
    device_name : str
        Name of the device that will be sending messages
    spacing : float
        Optional spacing to use for the received image. Overrides
        spacing included in message, if present.

    Returns
    -------
    None
    """

    if act_as_host:
        conn = pyigtl.OpenIGTLinkServer(port)
    else:
        conn = pyigtl.OpenIGTLinkClient(host, port)

    print("Waiting for connection")
    while not conn.is_connected():
        time.sleep(0.1)
    print("Connection established")

    while True:
        msg = conn.wait_for_message(device_name, timeout=-1)
        if msg is None:
            break
        if spacing is None:
            spacing = (msg.ijk_to_world_matrix[0, 0], msg.ijk_to_world_matrix[1, 1])

        img = msg.image
        if img.ndim == 3:
            img = img.squeeze()
        q.put((img, spacing), timeout=20)


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    Returns
    ----------
    argparse.ArgumentParser
    """

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument(
        "--no_plot_onsd",
        action="store_true",
        help="Don't plot the rolling onsd over time"
    )

    my_parser.add_argument(
        "--separate_source_and_mask",
        action="store_true",
        help="Make a separate widget for the source and mask frames."
        "Default behavior is to overlay the mask on the source image, all on"
        "the same widget"
    )

    my_parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=18944,
        help="Port for the server to run on",
    )

    my_parser.add_argument(
        "--model_path",
        action="store",
        type=Path,
        required=True,
        help="Path to the model to use for running the inference"
    )

    my_parser.add_argument(
        "--device_name",
        action="store",
        required=True,
        help="Device name from which the incoming messages will be sent"
    )

    my_parser.add_argument(
        "--act_as_host",
        action="store_true",
        help="Whether or not the gui should act like a client instead of a server"
    )

    # TODO check if above flag is present
    my_parser.add_argument(
        "--host",
        action="store",
        type=str,
        default="127.0.0.1"
    )

    my_parser.add_argument(
        "--spacing_override",
        action="store",
        type=float,
    )

    my_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Divide by 255 before adding image to the queue"
    )

    my_parser.add_argument(
        "--encoder_name",
        action="store",
        default="resnet34",
        help="Encoder name for the model to load. Defaults to resnet34"
    )

    return my_parser


if __name__ == "__main__":
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    # Force load itk dlls if we will need them.
    if not args.no_plot_onsd:
        print("Loading ITK libraries")
        itk.force_load()

    app = QApplication(sys.argv)
    q = Queue()
    host = None if args.act_as_host else args.host
    spacing = None
    if args.spacing_override:
        spacing = (args.spacing_override, args.spacing_override)

    p = Process(target=wait_and_add_image_to_queue, args=(q, args.port, args.act_as_host, host, args.device_name, spacing))
    p.start()

    # Let the python interpreter run every so often
    # https://stackoverflow.com/questions/4938723/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-co
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    w = MainWindow(
        q,
        args.model_path,
        plot_onsd=not args.no_plot_onsd,
        separate_source_and_mask=args.separate_source_and_mask,
        normalize=args.normalize,
        encoder_name=args.encoder_name,
    )
    w.show()
    app.exec_()
    p.terminate()
