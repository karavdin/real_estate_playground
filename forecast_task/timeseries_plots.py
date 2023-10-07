import warnings

import altair as alt
import pandas as pd
import numpy as np

import style as stl  # package

warnings.filterwarnings("ignore")

def enrich_day(df, time_col="C_DATE"):
    """
    Add time related features from time_col
    """
    df["ISODAY"] = df[time_col].dt.dayofyear
    df["WEEKDAY"] = df[time_col].dt.weekday.astype(np.int8)
    df["WEEKDAY_NAME"] = df[time_col].dt.day_name()
    #df["PREVIOUS_WEEK_SATURDAY"] = df[time_col] - pd.to_timedelta(2 + df["WEEKDAY"].values, "D")

    df["THIS_WEEK_MONDAY"] = df[time_col] - pd.to_timedelta(df["WEEKDAY"].values, "D")

    df["YEAR"] = pd.to_datetime(df[time_col]).dt.isocalendar().year
    df["WEEK"] = pd.to_datetime(df[time_col]).dt.isocalendar().week
    df["MONTH_DAY"] = pd.to_datetime(df[time_col]).dt.day
    df["MONTH"] = pd.to_datetime(df[time_col]).dt.month.astype(np.int8)
    df["WEEK_OF_MONTH"] = df[time_col].dt.day.astype(np.int8) // 7
    #df["TIMEDELTA_2000"] = ((df[time_col] - pd.to_datetime("2000-01-01")).dt.days / 7).astype(int)  # weeks, for trend
    df["TIMEDELTA"] = pow(
        ((df[time_col] - df[time_col].min()).dt.days / 7).astype(int), 1
    )  # weeks, for data weighting in training

    return df
    
def define_colors(var_cols, actuals_col, plot_last_year, plot_two_years_ago):
    """
    Define color scheme for the plotting
    Args:
        var_cols (List[str]): list of variables to plot in addition to actuals
        actuals_col (str): name of column with actuals
        plot_last_year (bool): flag to plot last year actuals
        plot_two_years_ago (book): flag to plot actuals from 2 years ago

    Returns:
        mycolors_map: map of colors
        stroke_dash_map: map of line types
    """
    stl.def_style()
    mycolors_map = {
        actuals_col: stl.mycolors_dict["Green"],
    }

    if len(var_cols) > 0:
        mycolors_map[var_cols[0]] = stl.mycolors_dict["Blue"]
    if len(var_cols) > 1:
        mycolors_map[var_cols[1]] = stl.mycolors_dict["Dark Blue"]
    if len(var_cols) > 2:
        mycolors_map[var_cols[2]] = stl.mycolors_dict["Purple"]
    if len(var_cols) > 3:
        mycolors_map[var_cols[3]] = stl.mycolors_dict["Orange"]
    if len(var_cols) > 4:
        mycolors_map[var_cols[4]] = stl.mycolors_dict["Magenta"]
    if len(var_cols) > 5:
        mycolors_map[var_cols[5]] = stl.mycolors_dict["Blue 2"]
    if len(var_cols) > 6:
        mycolors_map[var_cols[6]] = stl.mycolors_dict["Beigegrau"]
    if len(var_cols) > 7:
        mycolors_map[var_cols[7]] = stl.mycolors_dict["Pink"]
    if len(var_cols) > 8:
        mycolors_map[var_cols[8]] = stl.mycolors_dict["Dark Green"]
    if len(var_cols) > 9:
        mycolors_map[var_cols[9]] = stl.mycolors_dict["Black"]
    if len(var_cols) > 10:
        "You try to plot more than 10 columns at once. The plot will be messy. Please reduce number of columns!!"

    stroke_dash_map = {
        actuals_col: [0],
    }
    for i in range(len(var_cols)):
        stroke_dash_map[var_cols[i]] = [3, 3]
    if plot_last_year:
        mycolors_map[f"{actuals_col}_last_year"] = stl.mycolors_dict["Dark Gray"]
        stroke_dash_map[f"{actuals_col}_last_year"] = [0]
    if plot_two_years_ago:
        mycolors_map[f"{actuals_col}_-2_years"] = stl.mycolors_dict["Medium Light Gray"]
        stroke_dash_map[f"{actuals_col}_-2_years"] = [0]
    return mycolors_map, stroke_dash_map


def plot_pred_shipment_ts(
    df, xaxis=["c_date"], xaxis_label="date", lable="total sum", var_list=None, color_list=None, stroke_list=None
):
    """
    Plot timeseries
    """

    if "daily" or "weekly" in lable:
        xaxis_conv = "yearmonthdate(" + xaxis[0] + "):T"
    else:
        xaxis_conv = xaxis[0] + ":O"
    interval = alt.selection_interval(encodings=["x"])
    df_melted = pd.melt(df, id_vars=xaxis, value_vars=var_list).reset_index()
    chart = (
        alt.Chart(df_melted)
        .mark_line()
        .encode(
            x=alt.X(
                xaxis_conv,
                axis=alt.Axis(title=xaxis_label, labelAngle=-45),
            ),
            y=alt.Y("value:Q", axis=alt.Axis(title="QTY")),
            tooltip=[xaxis_conv, "variable", "value"],
            color=alt.Color(
                "variable",
                scale=alt.Scale(domain=var_list, range=color_list),
                legend=alt.Legend(title="", orient="right", symbolLimit=0, labelLimit=1000),
            ),
            strokeDash=alt.StrokeDash("variable", scale=alt.Scale(domain=var_list, range=stroke_list)),
        )
        .properties(title=lable)
        .add_selection(interval)
    )

    return chart


def add_last_year_weekly_actuals(df_total_aggregated, actuals_col):
    """
    Calculate actuals from one year ago
    """
    df_total_aggregated_week = pd.DataFrame()
    for year in df_total_aggregated.YEAR.unique():
        df_total_aggregated_last = df_total_aggregated[df_total_aggregated["YEAR"] == (year - 1)][["WEEK", actuals_col]]
        df_total_aggregated_last[f"{actuals_col}_last_year"] = df_total_aggregated_last[actuals_col]
        df_total_aggregated_last["current_year"] = year
        df_total_aggregated_last = df_total_aggregated_last.drop(columns=[actuals_col])
        df_total_aggregated_week = pd.concat([df_total_aggregated_week,
            df_total_aggregated[["YEAR", "WEEK", actuals_col]].merge(
                df_total_aggregated_last[["current_year", "WEEK", f"{actuals_col}_last_year"]],
                left_on=["YEAR", "WEEK"],
                right_on=["current_year", "WEEK"],
                how="inner",
            )]
        )
        df_total_aggregated_week = df_total_aggregated_week.drop(columns=["current_year"])
    return df_total_aggregated_week


def add_minus_2_year_weekly_actuals(df_total_aggregated, actuals_col):
    """
    Calculate actuals from 2 years ago
    """
    df_total_aggregated_week = pd.DataFrame()
    for year in df_total_aggregated.YEAR.unique():
        df_total_aggregated_last = df_total_aggregated[df_total_aggregated["YEAR"] == (year - 2)][["WEEK", actuals_col]]
        df_total_aggregated_last[f"{actuals_col}_-2_years"] = df_total_aggregated_last[actuals_col]
        df_total_aggregated_last["current_year"] = year
        df_total_aggregated_last = df_total_aggregated_last.drop(columns=[actuals_col])
        df_total_aggregated_week = pd.concat([df_total_aggregated_week,
            df_total_aggregated[["YEAR", "WEEK", actuals_col]].merge(
                df_total_aggregated_last[["current_year", "WEEK", f"{actuals_col}_-2_years"]],
                left_on=["YEAR", "WEEK"],
                right_on=["current_year", "WEEK"],
                how="inner",
            )]
        )
        df_total_aggregated_week = df_total_aggregated_week.drop(columns=["current_year"])
    return df_total_aggregated_week


def calc_agg_weekly(df, var_cols=["PREDICTIONS"], agg="sum", actuals_col="N_SALES"):
    """
    Calculate weekly aggregated data
    """
    agg_dict = {}
    for col in var_cols:
        agg_dict[col] = agg
    agg_dict[actuals_col] = agg
    agg_dict["THIS_WEEK_MONDAY"] = "first"

    df_total_aggregated = df.groupby(["WEEK", "YEAR"]).agg(agg_dict).reset_index()

    df_total_aggregated_week_lastyear = add_last_year_weekly_actuals(df_total_aggregated, actuals_col)
    df_total_aggregated = df_total_aggregated.merge(
        df_total_aggregated_week_lastyear[["YEAR", "WEEK", f"{actuals_col}_last_year"]],
        on=["WEEK", "YEAR"],
        how="outer",
    )
    df_total_aggregated_week_minus2 = add_minus_2_year_weekly_actuals(df_total_aggregated, actuals_col)
    df_total_aggregated = df_total_aggregated.merge(
        df_total_aggregated_week_minus2[["YEAR", "WEEK", f"{actuals_col}_-2_years"]], on=["WEEK", "YEAR"], how="outer"
    )
    return df_total_aggregated


def plot_weekly(
    df,
    lable="weekly total sum",
    agg="sum",
    var_cols=["PREDICTIONS"],
    actuals_col="N_SALES",
    date_from="2020-01-01",
    date_upto="2020-01-31",
    plot_last_year=True,
    plot_two_years_ago=True,
):
    """
    Plot weekly data per slice
    """
    mycolors_map, stroke_dash_map = define_colors(var_cols, actuals_col, plot_last_year, plot_two_years_ago)
    df = enrich_day(df)

    charts = []
    df_total_aggregated = calc_agg_weekly(df, var_cols=var_cols, agg=agg, actuals_col=actuals_col)
    var_list = []
    var_list += [actuals_col]
    if plot_last_year:
        var_list += [f"{actuals_col}_last_year"]
    if plot_two_years_ago:
        var_list += [f"{actuals_col}_-2_years"]
    var_list += var_cols

    color_list = []
    stroke_list = []
    for var in var_list:
        color_list.append(mycolors_map[var])
        stroke_list.append(stroke_dash_map[var])

    chart_ts = plot_pred_shipment_ts(
        df_total_aggregated[
            (df_total_aggregated["THIS_WEEK_MONDAY"] >= date_from)
            & (df_total_aggregated["THIS_WEEK_MONDAY"] < date_upto)
        ],
        xaxis=["THIS_WEEK_MONDAY"],
        xaxis_label="",
        lable=lable,
        var_list=var_list,
        color_list=color_list,
        stroke_list=stroke_list,
    )
    charts.append(chart_ts)

    chart_out = alt.concat(*charts, columns=1).configure_axis(grid=False)
    # chart_out.display()
    return chart_out


def plot_weekly_per_agg(
    df,
    actuals_col="N_SALES",
    var_cols=["Predictions"],
    agg_col="U_CHAIN",
    agg="sum",
    limit_n_plots=4,
    title_text="",
    date_from="2020-01-01",
    date_upto="2020-01-31",
    plot_last_year=True,
    plot_two_years_ago=True,
):
    """
    Plot aggregated weekly time series

    Parameters:
        df (DataFrame): input dataframe
        actuals_col (str): column with actuals
        val_cols (list(str)): other columns to plot on Y-axis (e.g predictions)
        agg_col (str): aggregation column (e.g product-group, location, etc)
        agg (str): aggregation function (only "sum" or "mean" are supported)
        limit_n_plots (str): number of plots to show. Plots are ordered by max(actuals_col.agg)
        title_text (str): text to be added to title
        date_from (str): first date on the plot, format 'YYYY-MM-DD'
        date_upto (str): last date on the plot, format 'YYYY-MM-DD'
        plot_last_year (bool): plot last year actuals
        plot_two_years_ago (bool): plot actuals from 2 years ago

    """
    var_list = (
        df.groupby([agg_col])
        .agg({actuals_col: agg})
        .sort_values(by=actuals_col, ascending=False)
        .reset_index()
        .head(limit_n_plots)[agg_col]
    )
    for var in var_list:
        df_slice = df[df[agg_col] == var]
        if df_slice[df_slice['C_DATE']>date_from].shape[0]>0:
            chart = plot_weekly(
                df_slice,
                lable=str(var) + f" weekly {agg}" + f" {title_text}",
                agg=agg,
                var_cols=var_cols,
                actuals_col=actuals_col,
                date_from=date_from,
                date_upto=date_upto,
                plot_last_year=plot_last_year,
                plot_two_years_ago=plot_two_years_ago,
            )
            chart.display()


def plot_daily(
    df,
    lable="daily total sum",
    agg="sum",
    var_cols=["PREDICTIONS"],
    actuals_col="N_SALES",
    date_from="2020-01-01",
    date_upto="2020-01-31",
    plot_last_year=True,
    plot_two_years_ago=True,
):
    """
    Plot daily data per slice
    """

    mycolors_map, stroke_dash_map = define_colors(var_cols, actuals_col, plot_last_year, plot_two_years_ago)

    charts = []
    df_total_aggregated = calc_agg_daily(df, var_cols=var_cols, agg=agg, actuals_col=actuals_col)
    var_list = []
    var_list += [actuals_col]
    if plot_last_year:
        var_list += [f"{actuals_col}_last_year"]
    if plot_two_years_ago:
        var_list += [f"{actuals_col}_-2_years"]
    var_list += var_cols

    color_list = []
    stroke_list = []
    for var in var_list:
        color_list.append(mycolors_map[var])
        stroke_list.append(stroke_dash_map[var])

    chart_ts = plot_pred_shipment_ts(
        df_total_aggregated[(df_total_aggregated["C_DATE"] >= date_from) & (df_total_aggregated["C_DATE"] < date_upto)],
        xaxis=["C_DATE"],
        xaxis_label="",
        lable=lable,
        var_list=var_list,
        color_list=color_list,
        stroke_list=stroke_list,
    )
    charts.append(chart_ts)

    chart_out = alt.concat(*charts, columns=1).configure_axis(grid=False)
    chart_out.display()
    # return chart_out


def add_last_year_daily_actuals(df_total_aggregated, actuals_col):
    """
    Calculate daily actuals from one year ago
    """
    df_total_aggregated_week = pd.DataFrame()
    for year in df_total_aggregated.YEAR.unique():
        df_total_aggregated_last = df_total_aggregated[df_total_aggregated["YEAR"] == (year - 1)][
            ["ISODAY", actuals_col]
        ]
        df_total_aggregated_last[f"{actuals_col}_last_year"] = df_total_aggregated_last[actuals_col]
        df_total_aggregated_last["current_year"] = year
        df_total_aggregated_last = df_total_aggregated_last.drop(columns=[actuals_col])
        df_total_aggregated_week = pd.concat([df_total_aggregated_week,
            df_total_aggregated[["YEAR", "ISODAY", actuals_col]].merge(
                df_total_aggregated_last[["current_year", "ISODAY", f"{actuals_col}_last_year"]],
                left_on=["YEAR", "ISODAY"],
                right_on=["current_year", "ISODAY"],
                how="inner",
            )]
        )
        df_total_aggregated_week = df_total_aggregated_week.drop(columns=["current_year"])
    return df_total_aggregated_week


def add_minus_2_year_daily_actuals(df_total_aggregated, actuals_col):
    """
    Calculate daily actuals from 2 years ago
    """
    df_total_aggregated_week = pd.DataFrame()
    for year in df_total_aggregated.YEAR.unique():
        df_total_aggregated_last = df_total_aggregated[df_total_aggregated["YEAR"] == (year - 2)][
            ["ISODAY", actuals_col]
        ]
        df_total_aggregated_last[f"{actuals_col}_-2_years"] = df_total_aggregated_last[actuals_col]
        df_total_aggregated_last["current_year"] = year
        df_total_aggregated_last = df_total_aggregated_last.drop(columns=[actuals_col])
        df_total_aggregated_week = pd.concat([df_total_aggregated_week,
            df_total_aggregated[["YEAR", "ISODAY", actuals_col]].merge(
                df_total_aggregated_last[["current_year", "ISODAY", f"{actuals_col}_-2_years"]],
                left_on=["YEAR", "ISODAY"],
                right_on=["current_year", "ISODAY"],
                how="inner",
            )]
        )
        df_total_aggregated_week = df_total_aggregated_week.drop(columns=["current_year"])
    return df_total_aggregated_week


def calc_agg_daily(df, var_cols=["PREDICTIONS"], agg="sum", actuals_col="N_SALES"):
    """
    Calculate daily aggregated data
    """
    agg_dict = {}
    for col in var_cols:
        agg_dict[col] = agg
    agg_dict[actuals_col] = agg
    agg_dict["C_DATE"] = "first"

    df_total_aggregated = df.groupby(["ISODAY", "YEAR"]).agg(agg_dict).reset_index()

    df_total_aggregated_week_lastyear = add_last_year_daily_actuals(df_total_aggregated, actuals_col)
    df_total_aggregated = df_total_aggregated.merge(
        df_total_aggregated_week_lastyear[["YEAR", "ISODAY", f"{actuals_col}_last_year"]],
        on=["ISODAY", "YEAR"],
        how="outer",
    )
    df_total_aggregated_week_minus2 = add_minus_2_year_daily_actuals(df_total_aggregated, actuals_col)
    df_total_aggregated = df_total_aggregated.merge(
        df_total_aggregated_week_minus2[["YEAR", "ISODAY", f"{actuals_col}_-2_years"]],
        on=["ISODAY", "YEAR"],
        how="outer",
    )
    return df_total_aggregated


def plot_daily_per_agg(
    df,
    actuals_col="N_SALES",
    var_cols=["Predictions"],
    agg_col="U_CHAIN",
    agg="sum",
    limit_n_plots=4,
    title_text="",
    date_from="2020-01-01",
    date_upto="2020-01-31",
    plot_last_year=True,
    plot_two_years_ago=True,
):
    """
    Plot aggregated daily time series

    Parameters:
        df (DataFrame): input dataframe
        actuals_col (str): column with actuals
        val_cols (list(str)): other columns to plot on Y-axis (e.g predictions)
        agg_col (str): aggregation column (e.g product-group, location, etc)
        agg (str): aggregation function (only "sum" or "mean" are supported)
        limit_n_plots (str): number of plots to show. Plots are ordered by max(actuals_col.agg)
        title_text (str): text to be added to title
        date_from (str): first date on the plot, format 'YYYY-MM-DD'
        date_upto (str): last date on the plot, format 'YYYY-MM-DD'
        plot_last_year (bool): plot last year actuals
        plot_two_years_ago (bool): plot actuals from 2 years ago

    """

    df = enrich_day(df)
    var_list = (
        df.groupby([agg_col])
        .agg({actuals_col: agg})
        .sort_values(by=actuals_col, ascending=False)
        .reset_index()
        .head(limit_n_plots)[agg_col]
    )
    for var in var_list:
        df_slice = df[df[agg_col] == var]
        plot_daily(
            df_slice,
            lable=str(var) + f" daily {agg}" + f" {title_text}",
            agg=agg,
            var_cols=var_cols,
            actuals_col=actuals_col,
            date_from=date_from,
            date_upto=date_upto,
            plot_last_year=plot_last_year,
            plot_two_years_ago=plot_two_years_ago,
        )
