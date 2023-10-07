import altair as alt
import numpy as np
import pandas as pd

import style as stl  # package


def get_metrics(df, pred_name="pred_mean_h30", target_name="target"):
    """
    Calculate metrics
    """
    df = df.dropna(subset=[target_name, pred_name])
    total_target = df[target_name].sum()
    total_predictions = df[pred_name].sum()
    n_products = df["n_Products"].values[0]
    df_stats = None

    if len(df) > 0:
        df["d"] = df[pred_name] - df[target_name]
        df["ad"] = np.abs(df[pred_name] - df[target_name])
        df["ad_neglect_zero_fc"] = np.abs(df[pred_name] - df[target_name])
        df.loc[df[pred_name] == 0, "ad_neglect_zero_fc"] = 0
        df["se"] = (df[pred_name] - df[target_name]) ** 2
        df["ad_norm"] = df["ad"] / df[target_name]
        mean_target = df[target_name].mean()
        mean_predictions = df[pred_name].mean()

        # sum prediction for days with no sales
        zero_pred = (df[df[target_name] == 0][pred_name].sum()) / total_predictions

        md = df["d"].mean()
        mad = df["ad"].mean()
        rmse = np.sqrt(df["se"].mean())

        if mean_target > 0:
            rmad = mad / mean_target
            dfa = 1 - rmad
        else:
            rmad = None
            dfa = 0

        target_sum = df[target_name].sum()
        pred_sum = df[pred_name].sum()
        # b1 = (df[pred_name] - df[target_name]) / df[pred_name]
        # gross_accuracy = (1 - np.abs(b1)).median()

        mask_0target = df[target_name] > 0
        qq = df[mask_0target].copy()
        wmape2 = sum(np.abs(qq[pred_name] - qq[target_name])) / qq[target_name].sum()

        bias = (pred_sum - target_sum) / target_sum
        net_accuracy = 1 + bias

        wmape = sum(df["ad"]) / target_sum
        fa_neglect_zero_fc = 1 - sum(df["ad_neglect_zero_fc"]) / target_sum

        # outlier metric used by Avon
        df["is_outlier"] = (-1) * df["d"] > df[pred_name]
        df["outlier_metric"] = 0
        df.loc[df["is_outlier"], "outlier_metric"] = (-1) * df["d"] - df[pred_name]
        outlier_kpi = df["outlier_metric"].sum() / target_sum

        df_stats = pd.DataFrame(
            [
                total_target,
                total_predictions,
                zero_pred,
                n_products,
                mean_target,
                mean_predictions,
                md,
                bias,
                rmad,
                rmse
            ]
        ).T

        df_stats.columns = [
            "Total Actuals",
            "Total Predictions",
            "% predictions for 0 actuals",
            "Categories",
            "Mean Actuals",
            "Mean Predictions",
            "MD",
            "Bias",
            "RMAD",
            "RMSE"
        ]

    return df_stats


def get_metrics_comp(
    df, pred_name_1="pred_mean_h30", pred_name_2="pred_mean_h30", pred_name_3=None, target_name="target"
):
    """
    Calculate metrics for 2 predictions
    """
    df1 = get_metrics(df=df, pred_name=pred_name_1, target_name=target_name)
    df1["fc_name"] = pred_name_1
    df2 = get_metrics(df=df, pred_name=pred_name_2, target_name=target_name)
    df2["fc_name"] = pred_name_2
    if pred_name_3 is None:
        df_stats_all = pd.concat([df1, df2])
    else:
        df3 = get_metrics(df=df, pred_name=pred_name_3, target_name=target_name)
        df3["fc_name"] = pred_name_3
        df_stats_all = pd.concat([df1, df2, df3])
    return df_stats_all


def color_negative(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: Orange'` for negative
    strings, Purple otherwise.
    """
    stl.def_style()

    color = stl.mycolors_dict["Orange"] if val < 0 else stl.mycolors_dict["Purple"]
    return "color: %s" % color


def return_kpis(df_kpi, caption, pred_from, pred_upto):
    """
    Nice formated KPI table
    """
    return (
        df_kpi.style.format(
            "{:.1%}",
            subset=[
                "RMAD",
                "% predictions for 0 actuals"
            ],
        )
        .format(
            "{0:,.0f}",
            subset=["Total Actuals", "Total Predictions", "Categories"],
        )
        .format("{0:,.2f}", subset=["MD", "Bias", "Mean Actuals", "Mean Predictions", "RMSE"])
        # .background_gradient(cmap="binary", subset=["Total Actuals"])
        .applymap(color_negative, subset=["MD", "Bias"])
        .set_caption(f"{caption} | predictions from {pred_from} to {pred_upto}")
    )


def KPI_per_agg_var_df(
    df,
    var="pg_name_2",
    limit=100,
    target_name="Quantity",
    pred_name="pred_mean_causal_h7",
    pred_name_comp=None,
    pred_name_comp_2=None,
    label=" ",
    granularity_list=["p_code", "U_CUSTOMER_AGGREGATION_NAME", "U_REGION_DESC_HIER", "THIS_WEEK_MONDAY"],
    pred_from="2022-01-01",
    pred_upto="2022-02-01",
    product_col=None,
    time_col="C_DATE",
    debug=False,
):
    """
    Dataframe with KPIs
    """
    df_calc = df[(df[time_col] >= pd.to_datetime(pred_from)) & (df[time_col] <= pd.to_datetime(pred_upto))]
    var_list = (
        df_calc.groupby([var])
        .agg({target_name: "sum"})
        .sort_values(by=target_name, ascending=False)
        .reset_index()
        .head(limit)[var]
    )
    df_stats_all = pd.DataFrame()
    for var_ in var_list:
        if debug:
            print("slice:", var_)
        df_slice = df_calc[(df_calc[var] == var_)]
        if len(df_slice) > 0:
            groupvars = granularity_list
            if pred_name_comp is None:
                df_PLC = df_slice.groupby(groupvars).agg({target_name: "sum", pred_name: "sum"}).reset_index()
            else:
                if pred_name_comp_2 is None:
                    df_PLC = (
                        df_slice.groupby(groupvars)
                        .agg({target_name: "sum", pred_name: "sum", pred_name_comp: "sum"})
                        .reset_index()
                    )
                else:
                    df_PLC = (
                        df_slice.groupby(groupvars)
                        .agg({target_name: "sum", pred_name: "sum", pred_name_comp: "sum", pred_name_comp_2: "sum"})
                        .reset_index()
                    )

            if product_col is None:
                df_PLC["n_Products"] = 0
            else:
                df_PLC["n_Products"] = len(df_slice[product_col].unique())
            if pred_name_comp is None:
                df_stats = get_metrics(df_PLC, pred_name=pred_name, target_name=target_name)
                df_stats[var] = var_
            else:
                if pred_name_comp_2 is None:
                    df_stats = get_metrics_comp(
                        df_PLC, pred_name_1=pred_name, pred_name_2=pred_name_comp, target_name=target_name
                    )
                else:
                    df_stats = get_metrics_comp(
                        df_PLC,
                        pred_name_1=pred_name,
                        pred_name_2=pred_name_comp,
                        pred_name_3=pred_name_comp_2,
                        target_name=target_name,
                    )
                df_stats[var] = df_stats["fc_name"] + " " + str(var_)
                # df_stats.drop(columns=["fc_name"], inplace=True)
            df_stats["category"] = var_
            # df_stats = df_stats.set_index([var])
            df_stats_all = pd.concat([df_stats_all,df_stats])
            df_stats_all = df_stats_all.sort_values(by="Total Actuals", ascending=False)
    if debug:
        print("df_stats_all.shape:", df_stats_all.shape)
    return df_stats_all


# get KPI per some level
def KPI_per_agg_var(
    df,
    var="pg_name_2",
    limit=100,
    target_name="Quantity",
    pred_name="pred_mean_causal_h7",
    pred_name_comp=None,
    pred_name_comp_2=None,
    label=" ",
    granularity_list=["p_code", "U_CUSTOMER_AGGREGATION_NAME", "U_REGION_DESC_HIER", "THIS_WEEK_MONDAY"],
    pred_from="2022-01-01",
    pred_upto="2022-02-01",
    product_col=None,
    time_col="C_DATE",
    debug=False,
):
    """
    Nicely formated table with KPIs per aggregated level
    """
    df_stats_all = KPI_per_agg_var_df(
        df=df,
        var=var,
        limit=limit,
        target_name=target_name,
        pred_name=pred_name,
        pred_name_comp=pred_name_comp,
        pred_name_comp_2=pred_name_comp_2,
        label=label,
        granularity_list=granularity_list,
        pred_from=pred_from,
        pred_upto=pred_upto,
        product_col=product_col,
        time_col=time_col,
        debug=debug,
    )
    if debug:
        print(df_stats_all.shape)
    df_stats_all = df_stats_all.set_index([var])
    df_stats_all.drop(columns=["category"], inplace=True)
    if pred_name_comp:
        df_stats_all.drop(columns=["fc_name"], inplace=True)
    df_stats_all_cl = return_kpis(df_stats_all, label, pred_from, pred_upto)
    return df_stats_all_cl


def plot_kpi_compare_per_agg(df_kpi, fc_1, fc_2, fc_3=None, kpi="RMAD", kpi_label="metric of choice"):
    """Visualise KPIs comparison from table"""
    mycolors_map_ = {fc_1: stl.mycolors_dict["Blue"], fc_2: stl.mycolors_dict["Dark Blue"]}
    var_list = [fc_1, fc_2]
    if fc_3:
        mycolors_map_[fc_3] = stl.mycolors_dict["Purple"]
        var_list.append(fc_3)
    color_list = []
    for var in var_list:
        color_list.append(mycolors_map_[var])

    df_cats = df_kpi[["category", "fc_name", kpi]]
    chart = (
        alt.Chart(df_cats)
        .mark_bar()
        .encode(
            x=alt.X("fc_name", axis=alt.Axis(title="", labelAngle=360), sort=var_list),
            y=alt.Y(kpi, axis=alt.Axis(title=kpi_label)),
            column=alt.Column("category:N"),
            color=alt.Color("fc_name", scale=alt.Scale(domain=var_list, range=color_list), legend=None),
        )
        .configure_axis(grid=False)
        .properties(width=300, height=300)
    )
    # chart.display()
    return chart
