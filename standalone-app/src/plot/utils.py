import matplotlib.pyplot as plt

class PlotPickable():
    fig = None
    ax = None
    lined = {}

    def __init__ (self):
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot()

    def get_plot_props (self):
        return self.fig, self.ax

    def get_lined (self):
        return self.lined

    def define_legend_items (self, legend, plot_lines, plot_markers=None):
        if not plot_markers:
            for legline, origline in zip(legend.get_lines(), plot_lines):
                legline.set_picker(True)  # Enable picking on the legend line.
                self.lined[legline] = origline
        else:
            for legline, origline, origmark in zip(legend.get_lines(), plot_lines, plot_markers):
                legline.set_picker(True)  # Enable picking on the legend line.
                self.lined[legline] = (origline, origmark)
    
    @staticmethod
    def change_legend_alpha (legend):
        for legline in legend.get_lines():
            legline.set_alpha(1)

    @staticmethod
    def on_pick(event, fig, lined):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        try:
            origline, origmark = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            origmark.set_visible(visible)
        except TypeError:
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()