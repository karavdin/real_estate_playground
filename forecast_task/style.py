import altair as alt

mycolors_dict = {
    "Blue": "#00B7F1",
    "Dark Blue": "#000E4E",
    "Green": "#81BB41",
    "Purple": "#993399",
    "Orange": "#E75424",
    "Medium Light Gray": "#E6E7E8",
    "Light Gray": "#F1F1F2",
    "Dark Gray": "#BEC1C3",
    "White": "#FFFFFF",
    "Black": "#000000",
    "Magenta": "#f653a6",
    "Beigegrau": "#756f61",
    "Blue 2": "#0089b5",
    "Pink": "#c460aa",
    "Dark Green": "#263813",
}


def constantia():
    """
    Configure font and font sizes
    """
    font = "Constantia"

    return {
        "config": {
            "title": {"font": font, "fontSize": 24},
            "axis": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 24,
                "titleFontSize": 26,
            },
            "header": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 24,
                "titleFontSize": 26,
            },
            "legend": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 24,
                "titleFontSize": 26,
            },
            "view": {
                "height": 300,
                "width": 600,
            },
        }
    }


def def_style():
    """
    Define font and some other altair settings
    """

    alt.renderers.enable("default")
    alt.data_transformers.disable_max_rows()
    alt.themes.register("constantia", constantia)
    alt.themes.enable("constantia")
    # print("BY Style applied")
