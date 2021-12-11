
from PyQt5.QtGui import QColor


class Ui_handler:
    def __init__(self, mainwindow, app) -> None:
        self.app = app
        self.ui = mainwindow 

        self.init_event_listeners()

    def init_event_listeners(self):
        self.ui.btn_fetchFromArchiver.clicked.connect(self.app.on_btn_fetchFromArchiver_clicked)
        self.ui.btn_plot.clicked.connect(self.app.on_btn_plot_clicked)
        self.ui.btn_cleanData.clicked.connect(self.app.clean_loaded_data)
        self.ui.btn_makeVideo.clicked.connect(self.app.make_video)
        self.ui.check_selectPvs.toggled.connect(self.toggle_pv_input)

    @property
    def optimize(self) -> bool:
        return self.ui.check_optimize.isChecked()
    
    @property
    def time_in_minutes(self) -> int:
        return self.ui.spin_timeOptimize.value()

    @property
    def filter_max(self) -> float:
        return self.ui.spin_filter_max.value()
    
    @property
    def filter_min(self) -> float:
        return self.ui.spin_filter_min.value()

    @property
    def filter_data(self) -> bool:
        return self.ui.check_applyFilter.isChecked()

    @property
    def figures_path(self) -> str:
        return self.ui.inputTxt_dirFig.text()

    @property
    def save_fig(self) -> bool:
        return self.ui.check_saveFig.isChecked()

    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: ui function to trigger 'enable' state of a textbox
        -------------------------------------------------------------------------------------------------------------------- """
    def toggle_pv_input(self):
        self.ui.inputTxt_pvs.setEnabled(self.ui.check_selectPvs.isChecked())

    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: ui function to display messages into text element
    Args:
        message: text to be displayed
        severity: indicator of the type of the message
            values: 'normal', 'danger', 'alert' or 'success'
        -------------------------------------------------------------------------------------------------------------------- """
    def logMessage(self, message, severity='normal'):
        if (severity != 'normal'):
            if (severity == 'danger'):
                color = 'red'
            elif (severity == 'alert'):
                color = 'yellow'
            elif (severity == 'success'):
                color = 'green'

            # saving properties
            tc = self.ui.log.textColor()
            self.ui.log.setTextColor(QColor(color))
            self.ui.log.append(message)
            self.ui.log.setTextColor(tc)
        else:
            self.ui.log.append(message)

    def update_pvs_loaded(self, pv: str) -> None:
        """ ui function to display a pv that was loaded """
        
        self.ui.txt_loadedPvs.append('▶ '+ pv)

    def get_timespam_formatted(self) -> dict:
        """ reads datetime inputs from UI and formats it in ISO (and Archiver) standard """

        # get datetime from ui
        date_ini = self.ui.datetime_init.date().toString('yyyy-MM-dd')
        time_ini = self.ui.datetime_init.time().addSecs(3*60*60).toString('THH:mm:ss.zzzZ')
        date_end = self.ui.datetime_end.date().toString('yyyy-MM-dd')
        time_end = self.ui.datetime_end.time().addSecs(3*60*60).toString('THH:mm:ss.zzzZ')
        if (self.ui.datetime_end.time().hour() >= 21):
            date_end = self.ui.datetime_end.date().addDays(1).toString('yyyy-MM-dd')
        
        return {'init': date_ini + time_ini, 'end': date_end + time_end}

    def get_pv_option(self) -> str:
        """ indicates which PV option the user selected """

        if (self.ui.check_allPvs.isChecked()):
            # column_names = self.HLS_LEGEND
            return 'hls_all'
        if (self.ui.check_rfPv.isChecked()):
            return 'rf'
        if (self.ui.check_wellpressure.isChecked()):
            return 'well'
        if (self.ui.check_earthtides.isChecked()):
            return 'tides'
        if (self.ui.check_selectPvs.isChecked()):
            return 'select'
        if (self.ui.check_opositeHLS.isChecked()):
            # column_names = ['HLS Easth-West', 'HLS North-South']
            # special_case = 'oposite HLS'
            return 'hls_oposite'

    def enable_actions(self) -> None:
        self.ui.btn_plot.setEnabled(True)
        self.ui.btn_makeVideo.setEnabled(True)
        self.ui.label_dataLoaded.setStyleSheet("background-color:rgb(163, 255, 138);color:rgb(0, 0, 0);padding:3;")
        self.ui.label_dataLoaded.setText("Data loaded")

    def get_plot_type(self) -> dict:
        
        # analysis
        if (self.ui.check_plotCorrel.isChecked()):
            analysis = 'correlation'
        elif (self.ui.check_plotDirectional.isChecked()):
            analysis = 'directional'
        elif (self.ui.check_plotFFT.isChecked()):
            analysis = 'fft'
        else:
            analysis = 'timeseries'
        
        # time: one (static) or multiples (dynamic) periods 
        if (self.ui.check_plotDynamic.isChecked()):
            static = False
        else:
            static = True

        # visual: 2D or 3D (applicable for a few cases)
        if (self.ui.check_plot2D.isChecked()):
            is_2d = True
        else:
            is_2d = False

        return {'analysis': analysis, 'is_static': static, 'is_2d': is_2d}